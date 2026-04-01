-- Add termination key (star/pound) columns for NPI and Tax ID DTMF input
ALTER TABLE insurance_providers ADD COLUMN IF NOT EXISTS ivr_npi_suffix VARCHAR(5) DEFAULT NULL;
ALTER TABLE insurance_providers ADD COLUMN IF NOT EXISTS ivr_tax_id_suffix VARCHAR(5) DEFAULT NULL;

-- Constrain suffix values to valid DTMF termination keys
DO $$ BEGIN
    ALTER TABLE insurance_providers ADD CONSTRAINT chk_ivr_npi_suffix
        CHECK (ivr_npi_suffix IS NULL OR ivr_npi_suffix IN ('*', '#'));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    ALTER TABLE insurance_providers ADD CONSTRAINT chk_ivr_tax_id_suffix
        CHECK (ivr_tax_id_suffix IS NULL OR ivr_tax_id_suffix IN ('*', '#'));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
