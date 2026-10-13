-- Core multi-tenant schema: organizations, workspaces, membership, documents, chunks,
-- access groups, API keys, audit log, background ingestion jobs.
-- Assumes auth.users / auth.uid() already exist (Supabase Auth in production;
-- db/local_shim.sql provides an equivalent for local/CI postgres).

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE organizations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE workspaces (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id uuid NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    name text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE memberships (
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    workspace_id uuid NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    role text NOT NULL DEFAULT 'member' CHECK (role IN ('admin', 'member', 'viewer')),
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, workspace_id)
);

CREATE TABLE access_groups (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id uuid NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    name text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, name)
);

CREATE TABLE access_group_members (
    group_id uuid NOT NULL REFERENCES access_groups(id) ON DELETE CASCADE,
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    PRIMARY KEY (group_id, user_id)
);

-- One row per uploaded file. content_hash dedupes re-uploads within a workspace;
-- ingestion (Phase 1b) replaces an existing document's chunks on a hash change
-- rather than creating a duplicate row.
CREATE TABLE documents (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id uuid NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    access_group_id uuid REFERENCES access_groups(id) ON DELETE SET NULL,
    filename text NOT NULL,
    content_type text NOT NULL,
    storage_path text NOT NULL,
    content_hash text NOT NULL,
    status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'processing', 'ready', 'failed')),
    error text,
    uploaded_by uuid REFERENCES auth.users(id),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, content_hash)
);

-- access_group_id is denormalized from documents.access_group_id at ingestion time so the
-- RLS policy below can filter chunks without a join back to documents on every query.
CREATE TABLE chunks (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    workspace_id uuid NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    access_group_id uuid REFERENCES access_groups(id) ON DELETE SET NULL,
    chunk_index int NOT NULL,
    page_number int,
    section_heading text,
    content text NOT NULL,
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding vector(384),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX chunks_workspace_idx ON chunks (workspace_id);
CREATE INDEX chunks_document_idx ON chunks (document_id);
CREATE INDEX chunks_tsv_idx ON chunks USING gin (content_tsv);
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE api_keys (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id uuid NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    name text NOT NULL,
    hashed_key text NOT NULL UNIQUE,
    created_by uuid REFERENCES auth.users(id),
    created_at timestamptz NOT NULL DEFAULT now(),
    last_used_at timestamptz,
    revoked_at timestamptz
);

CREATE TABLE audit_log (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id uuid REFERENCES workspaces(id) ON DELETE CASCADE,
    actor_id uuid REFERENCES auth.users(id),
    action text NOT NULL,
    resource_type text NOT NULL,
    resource_id uuid,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX audit_log_workspace_idx ON audit_log (workspace_id, created_at DESC);

-- Background ingestion queue (Phase 1b worker polls this). Internal only: no RLS
-- policies are granted to regular users below, so it is reachable only via the
-- service-role connection the worker/backend uses.
CREATE TABLE jobs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id uuid NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    document_id uuid REFERENCES documents(id) ON DELETE CASCADE,
    job_type text NOT NULL,
    status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'processing', 'done', 'failed')),
    attempts int NOT NULL DEFAULT 0,
    error text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX jobs_status_idx ON jobs (status, created_at);

-- ---- Grants: mirrors Supabase's own authenticated/service_role split so RLS is the
-- only thing standing between an authenticated user and another workspace's rows. ----

GRANT USAGE ON SCHEMA public TO authenticated, service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO service_role;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO authenticated, service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO authenticated;

-- ---- Row-level security: every tenant table filtered by workspace_id ----

ALTER TABLE workspaces ENABLE ROW LEVEL SECURITY;
ALTER TABLE memberships ENABLE ROW LEVEL SECURITY;
ALTER TABLE access_groups ENABLE ROW LEVEL SECURITY;
ALTER TABLE access_group_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;

CREATE POLICY own_memberships ON memberships
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY member_workspaces ON workspaces
    FOR SELECT USING (id IN (SELECT workspace_id FROM memberships WHERE user_id = auth.uid()));

CREATE POLICY member_select_access_groups ON access_groups
    FOR SELECT USING (workspace_id IN (SELECT workspace_id FROM memberships WHERE user_id = auth.uid()));
CREATE POLICY admin_insert_access_groups ON access_groups
    FOR INSERT WITH CHECK (workspace_id IN (SELECT workspace_id FROM memberships WHERE user_id = auth.uid() AND role = 'admin'));
CREATE POLICY admin_update_access_groups ON access_groups
    FOR UPDATE USING (workspace_id IN (SELECT workspace_id FROM memberships WHERE user_id = auth.uid() AND role = 'admin'));
CREATE POLICY admin_delete_access_groups ON access_groups
    FOR DELETE USING (workspace_id IN (SELECT workspace_id FROM memberships WHERE user_id = auth.uid() AND role = 'admin'));

CREATE POLICY read_group_membership ON access_group_members
    FOR SELECT USING (
        user_id = auth.uid()
        OR group_id IN (
            SELECT id FROM access_groups WHERE workspace_id IN (
                SELECT workspace_id FROM memberships WHERE user_id = auth.uid() AND role = 'admin'
            )
        )
    );
CREATE POLICY admin_manage_group_membership ON access_group_members
    FOR ALL USING (
        group_id IN (
            SELECT id FROM access_groups WHERE workspace_id IN (
                SELECT workspace_id FROM memberships WHERE user_id = auth.uid() AND role = 'admin'
            )
        )
    )
    WITH CHECK (
        group_id IN (
            SELECT id FROM access_groups WHERE workspace_id IN (
                SELECT workspace_id FROM memberships WHERE user_id = auth.uid() AND role = 'admin'
            )
        )
    );

-- A document in an access group is visible only to members of that group (admins always see everything).
CREATE POLICY tenant_select_documents ON documents
    FOR SELECT USING (
        workspace_id IN (SELECT workspace_id FROM memberships WHERE user_id = auth.uid())
        AND (
            access_group_id IS NULL
            OR access_group_id IN (SELECT group_id FROM access_group_members WHERE user_id = auth.uid())
            OR workspace_id IN (SELECT workspace_id FROM memberships WHERE user_id = auth.uid() AND role = 'admin')
        )
    );
CREATE POLICY member_insert_documents ON documents
    FOR INSERT WITH CHECK (workspace_id IN (SELECT workspace_id FROM memberships WHERE user_id = auth.uid() AND role IN ('admin', 'member')));
CREATE POLICY member_update_documents ON documents
    FOR UPDATE USING (workspace_id IN (SELECT workspace_id FROM memberships WHERE user_id = auth.uid() AND role IN ('admin', 'member')));
CREATE POLICY member_delete_documents ON documents
    FOR DELETE USING (workspace_id IN (SELECT workspace_id FROM memberships WHERE user_id = auth.uid() AND role IN ('admin', 'member')));

-- Chunks are written only by the service-role ingestion worker; this policy only governs
-- reads from authenticated API requests (retrieval queries).
CREATE POLICY tenant_select_chunks ON chunks
    FOR SELECT USING (
        workspace_id IN (SELECT workspace_id FROM memberships WHERE user_id = auth.uid())
        AND (
            access_group_id IS NULL
            OR access_group_id IN (SELECT group_id FROM access_group_members WHERE user_id = auth.uid())
            OR workspace_id IN (SELECT workspace_id FROM memberships WHERE user_id = auth.uid() AND role = 'admin')
        )
    );

CREATE POLICY admin_manage_api_keys ON api_keys
    FOR ALL USING (workspace_id IN (SELECT workspace_id FROM memberships WHERE user_id = auth.uid() AND role = 'admin'))
    WITH CHECK (workspace_id IN (SELECT workspace_id FROM memberships WHERE user_id = auth.uid() AND role = 'admin'));

CREATE POLICY admin_read_audit_log ON audit_log
    FOR SELECT USING (workspace_id IN (SELECT workspace_id FROM memberships WHERE user_id = auth.uid() AND role = 'admin'));

-- jobs: no policies for `authenticated` -> default-deny. Only service_role (BYPASSRLS) touches it.
