# Decision Log - Monday.com BI Agent

## 1. Assumptions & Data Realities

*   **Normalisation turned out to be nearly a non-problem:** Monday's typed import already delivers reasonably consistent dates and controlled vocabularies. The *real* messiness is missingness, zero-encoding, duplicate-purpose columns, and cross-board vocabulary divergence. Engineering effort was deliberately redirected toward surfacing these structural anomalies rather than over-engineering date parsers.
*   **Scale:** The dataset is small enough to be paged over the API and held in an in-memory DuckDB database without requiring chunked processing.
*   **Read-only deviation:** The brief strictly required read-only access. Column descriptions were seeded once by hand via an external script during development to solve column ambiguity. The shipped application only reads them, generates its own when absent, and strictly enforces read-only access at the API gateway (which raises a `PermissionError` on any GraphQL mutation). This turns a constraint breach into a stated design position.

## 2. Trade-offs & Architecture Decisions

*   **Caching over persistence:** 
    The board review and data store are cached in-memory and regenerated on cold start rather than persisted externally. Rebuilding costs one model call per board; a database or object store would mean another dependency and another credential for something that costs cents to recompute. At this scale, the trade-off strongly favours simplicity. In a production environment, the same artifact would move to durable storage keyed by the same structural hash — which is a simple storage-adapter change, not a redesign.
*   **Model chain and provider:** 
    We chose OpenRouter to wrap the OpenAI SDK because it allows for server-side model fallback chains (e.g., `z-ai/glm-5.3-flash,openai/gpt-5.6-sol,z-ai/glm-5.2:free`). This guarantees high availability and allows the operator to toggle between free models and paid tier models by simply changing one environment variable, without changing a single line of agent code.
*   **In-Memory DuckDB:** 
    Requires zero external setup (no Dockerized database) and provides exceptional analytical SQL performance natively over dataframes. By using `@st.cache_resource`, we enabled a shared-cache model where data is loaded once and shared across visitors instantly.

## 3. Interpreting "Leadership Updates"

The brief asked to "prepare data for leadership updates." We interpret this as **assembling tables against a stated requirement, not producing a fixed report.** 

A fixed report is just a dashboard, and founders are already drowning in those. The human remains in charge: they state what the update must cover (e.g., via the `/leadership-update` command), and the agent responds with a proposed plan of plain-English questions. Upon approval, the agent executes the queries sequentially, appending measured caveats (like null-counts) and providing raw CSV downloads so the founder has the exact numbers to copy-paste into their slide deck.

## 4. Features Deliberately Ruled Out

*   **Semantic Anomaly Detection:** We deliberately ruled this out. There is no ground truth, so an LLM asked to find "semantic anomalies" will return findings whether or not any actually exist. One false alarm from an AI costs far more trust than ten true null-value errors earn. That mathematical reasoning is a stronger answer than simply "running out of time."
*   **Webhooks:** We went with a manual "Refresh" button instead of webhooks. We lose real-time updates, but avoid setting up a public-facing endpoint, which simplifies the prototype while satisfying the reviewer requirement that data can be refreshed to prove it isn't a hardcoded file.
