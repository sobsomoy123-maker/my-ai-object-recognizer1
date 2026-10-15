# Netlify Functions

These functions use `@netlify/blobs` with named stores. On Netlify, the platform supplies the Blobs context for deployed functions. For local development, link the site with `netlify link` or run `netlify dev` after authenticating so Blobs can use the linked site's storage. No custom storage bucket URL is required.
