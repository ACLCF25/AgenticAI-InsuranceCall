---
name: Audio/Media Auth Pattern
description: Never use a direct URL in <audio src> for protected endpoints — always fetch as blob via axios
type: feedback
---

Do not use `<audio src={apiUrl}>` for backend endpoints protected by JWT (`@admin_required`). The browser audio element cannot attach the Authorization header, causing silent 401 failures.

**Why:** The `/api/call-recording/:id/stream` endpoint requires a Bearer token. A plain `<audio src="http://...">` request bypasses axios and gets a 401.

**How to apply:** Whenever rendering audio (or any media) from a protected backend route, use `api.getCallRecordingBlob(id)` (or similar axios call with `responseType: 'blob'`), create a blob URL via `URL.createObjectURL()`, use that as the `src`, and revoke the URL on component unmount.
