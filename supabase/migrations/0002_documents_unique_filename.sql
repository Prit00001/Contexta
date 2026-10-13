-- Dedupe/replace (ingestion pipeline, Phase 1b) needs document identity keyed on
-- (workspace_id, filename): re-uploading the same filename with new bytes replaces
-- that document's chunks; re-uploading identical bytes is a no-op. 0001 keyed
-- uniqueness on content_hash instead, which made two different files with the same
-- name create two rows rather than one being "replaced". Fix the constraint here
-- rather than editing an already-applied migration.

ALTER TABLE documents DROP CONSTRAINT documents_workspace_id_content_hash_key;
ALTER TABLE documents ADD CONSTRAINT documents_workspace_id_filename_key UNIQUE (workspace_id, filename);
