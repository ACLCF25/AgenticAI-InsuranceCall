-- Migration: Fix twilio_numbers.added_by FK to reference auth.users instead of public.users
-- Run this against your Supabase PostgreSQL database to repair existing deployments
-- where the table was created with the incorrect REFERENCES users(id) constraint.

DO $$
DECLARE
    constraint_name TEXT;
BEGIN
    SELECT tc.constraint_name INTO constraint_name
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
      ON tc.constraint_name = kcu.constraint_name
    WHERE tc.table_name = 'twilio_numbers'
      AND kcu.column_name = 'added_by'
      AND tc.constraint_type = 'FOREIGN KEY';

    IF constraint_name IS NOT NULL THEN
        EXECUTE 'ALTER TABLE twilio_numbers DROP CONSTRAINT ' || quote_ident(constraint_name);
    END IF;
END$$;

ALTER TABLE twilio_numbers
    ADD CONSTRAINT twilio_numbers_added_by_fkey
    FOREIGN KEY (added_by) REFERENCES auth.users(id) ON DELETE SET NULL;
