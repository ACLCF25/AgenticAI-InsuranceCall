-- Fix Optum IVR data so exact Optum calls do not rely on placeholder phrases
-- or UI-prefixed speech values.

UPDATE insurance_providers
SET ivr_asks_tax_id = TRUE,
    ivr_tax_id_method = 'dtmf',
    ivr_tax_id_digits_to_send = NULL,
    ivr_tax_id_suffix = NULL,
    last_updated = NOW()
WHERE lower(trim(insurance_name)) = 'optum'
  AND regexp_replace(COALESCE(phone_number, ''), '\D', '', 'g') = '8776140484';

UPDATE ivr_knowledge
SET detected_phrase = 'reason for your call',
    preferred_action = 'speech',
    action_value = 'Credentialing',
    last_updated = NOW()
WHERE lower(trim(insurance_name)) = 'optum'
  AND menu_level = 1;

UPDATE ivr_knowledge
SET detected_phrase = 'join the network',
    preferred_action = 'speech',
    action_value = 'Join the network',
    last_updated = NOW()
WHERE lower(trim(insurance_name)) = 'optum'
  AND (
      menu_level = 2
      OR lower(COALESCE(action_value, '')) LIKE '%join the network%'
  );
