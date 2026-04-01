-- Supabase Auth + role-based access upgrade
-- Run this against an existing database before deploying the new auth flow.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role') THEN
        CREATE TYPE user_role AS ENUM ('super_admin', 'admin', 'agent');
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'approval_status') THEN
        CREATE TYPE approval_status AS ENUM ('pending', 'approved', 'rejected');
    END IF;
END$$;

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    username VARCHAR(50) NOT NULL UNIQUE,
    role user_role NOT NULL DEFAULT 'agent',
    approval_status approval_status NOT NULL DEFAULT 'pending',
    approved_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    approved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_profiles_role ON user_profiles(role);
CREATE INDEX IF NOT EXISTS idx_user_profiles_approval_status ON user_profiles(approval_status);

ALTER TABLE credentialing_requests
    ADD COLUMN IF NOT EXISTS initiated_by UUID REFERENCES auth.users(id) ON DELETE SET NULL;

CREATE OR REPLACE FUNCTION public.handle_new_auth_user()
RETURNS TRIGGER AS $$
DECLARE
    requested_username TEXT;
BEGIN
    requested_username := NULLIF(TRIM(COALESCE(NEW.raw_user_meta_data ->> 'username', '')), '');
    IF requested_username IS NULL THEN
        requested_username := split_part(NEW.email, '@', 1);
    END IF;

    INSERT INTO public.user_profiles (user_id, username, role, approval_status)
    VALUES (NEW.id, requested_username, 'agent', 'pending')
    ON CONFLICT (user_id) DO NOTHING;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
AFTER INSERT ON auth.users
FOR EACH ROW
EXECUTE FUNCTION public.handle_new_auth_user();
