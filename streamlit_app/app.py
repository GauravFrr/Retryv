import streamlit as st
import httpx
import pandas as pd
import traceback

# Configure Page
st.set_page_config(
    page_title="Retryv — RAG Portfolio Demo",
    page_icon="⚡",
    layout="wide",
)

# Custom Styles
st.markdown(
    """
    <style>
    .accent { color: #7C3AED; font-weight: bold; }
    .success-badge {
        background-color: #DCFCE7;
        color: #16A34A;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .warning-badge {
        background-color: #FEF3C7;
        color: #D97706;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .metric-card {
        background-color: #1E1E2E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #313244;
        text-align: center;
    }
    .metric-card h4 {
        margin: 0 !important;
        color: #A6ADC8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    .metric-card p {
        margin: 8px 0 0 0 !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #CBA6F7 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# API Endpoint Configuration
API_BASE_URL = "http://localhost:8000/api/v1"


def check_backend():
    try:
        response = httpx.get(f"{API_BASE_URL.replace('/api/v1', '')}/health", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


# Header Section
st.title("⚡ Retryv — Production-Grade RAG Demo")
st.markdown("Ask questions over FastAPI's documentation with hybrid search and citation verification.")

# Check Backend Status
backend_ok = check_backend()
if not backend_ok:
    st.error(
        "⚠️ FastAPI backend service is not running. Please start the API server using:\n"
        "```powershell\n"
        "uvicorn app.main:app --reload\n"
        "```"
    )
    st.stop()

# Sidebar Controls
st.sidebar.header("RAG Configuration")
strategy = st.sidebar.selectbox(
    "Chunking Strategy",
    options=["fixed_size", "structure_aware", "semantic"],
    index=1,
    help="Determines how documents are segmented prior to indexing.",
)
method = st.sidebar.selectbox(
    "Retrieval Method",
    options=["hybrid", "dense", "sparse"],
    index=0,
    help="Method used to fetch relevant chunks.",
)
rerank = st.sidebar.checkbox(
    "Apply Cross-Encoder Reranker",
    value=True,
    disabled=(method != "hybrid"),
    help="If enabled, ms-marco-MiniLM-L-6-v2 will re-order fused candidates.",
)
verify_citations = st.sidebar.checkbox(
    "Verify Citations",
    value=True,
    help="If enabled, Gemini will verify that each cited chunk supports the generated answer.",
)

# Sidebar metadata / statistics
st.sidebar.markdown("---")
st.sidebar.subheader("Index Statistics")
try:
    stats_resp = httpx.get(f"{API_BASE_URL}/documents", timeout=5.0).json()
    if stats_resp.get("status") == "success":
        chroma_stats = stats_resp["chromadb_indexes"].get(strategy, {})
        st.sidebar.metric("Chunks in Database", chroma_stats.get("chunk_count", 0))
        bm25_exists = stats_resp["bm25_indexes"].get(strategy, {}).get("exists", False)
        st.sidebar.markdown(
            f"BM25 Index: {'✅ Present' if bm25_exists else '❌ Missing'}"
        )
except Exception:
    st.sidebar.warning("Failed to load database stats.")


# Tabs Definition
tab_ask, tab_compare, tab_eval, tab_status = st.tabs(
    ["Ask Q&A", "Compare Configurations", "Evaluation Report", "Index Status"]
)

# ==========================================
# 1. Ask Q&A Tab
# ==========================================
with tab_ask:
    st.subheader("Interactive Q&A Session")
    question = st.text_input(
        "Enter your question about FastAPI:",
        placeholder="How do lifespan events work in FastAPI?",
    )

    if st.button("Ask Question", type="primary") and question:
        payload = {
            "query": question,
            "strategy": strategy,
            "method": method,
            "rerank": rerank if method == "hybrid" else False,
            "verify_citations": verify_citations,
        }

        with st.spinner("Retrieving contexts and generating answer..."):
            try:
                response = httpx.post(f"{API_BASE_URL}/ask", json=payload, timeout=120.0)
                if response.status_code == 200:
                    data = response.json()
                    gen_data = data["generation"]

                    # Display Answer
                    st.markdown("### Answer")
                    st.write(gen_data["answer"])

                    # Display Metrics Row
                    st.markdown("### Metrics")
                    col_grounded, col_citations, col_ratio = st.columns(3)
                    with col_grounded:
                        st.markdown(
                            f"<div class='metric-card'><h4>Grounded</h4>"
                            f"<p>{'✅ Yes' if gen_data['is_grounded'] else '❌ No / Insufficient Context'}</p></div>",
                            unsafe_allow_html=True,
                        )
                    with col_citations:
                        st.markdown(
                            f"<div class='metric-card'><h4>Citations Generated</h4>"
                            f"<p>{len(gen_data['cited_chunks'])} Chunks</p></div>",
                            unsafe_allow_html=True,
                        )
                    with col_ratio:
                        if verify_citations and gen_data["is_grounded"]:
                            ratio = data["support_ratio"]
                            st.markdown(
                                f"<div class='metric-card'><h4>Citation Support Ratio</h4>"
                                f"<p>{ratio * 100:.1f}% ({'Fully Supported' if data['all_supported'] else 'Has Unsupported Claims'})</p></div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                "<div class='metric-card'><h4>Citation Support Ratio</h4><p>N/A (Verification Off)</p></div>",
                                unsafe_allow_html=True,
                            )

                    # Display Sources / Citations
                    st.markdown("### Reference Sources")
                    if gen_data["cited_chunks"]:
                        for idx, chunk in enumerate(gen_data["cited_chunks"]):
                            with st.expander(
                                f"[{chunk['index']}] {chunk['source_file']} — {chunk['section_heading']}"
                            ):
                                # If verified, show verdict
                                if verify_citations and idx < len(data["verifications"]):
                                    verdict = data["verifications"][idx]
                                    badge_class = (
                                        "success-badge"
                                        if verdict["supported"]
                                        else "warning-badge"
                                    )
                                    st.markdown(
                                        f"**Verification Verdict:** <span class='{badge_class}'>{verdict['verdict']}</span>",
                                        unsafe_allow_html=True,
                                    )
                                    if not verdict["supported"]:
                                        st.caption(
                                            "⚠️ The LLM Fact-Checker detected that the answer makes assertions not supported by this chunk."
                                        )

                                st.markdown("**Chunk Content:**")
                                st.write(chunk["text"])
                    else:
                        st.info("No sources were cited for this response.")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Failed to fetch response: {str(e)}")
                st.code(traceback.format_exc())

# ==========================================
# 2. Compare Tab
# ==========================================
with tab_compare:
    st.subheader("Side-by-Side Configuration Comparison")
    compare_question = st.text_input(
        "Enter question to compare configurations:",
        key="compare_q",
        placeholder="How do lifespan events work in FastAPI?",
    )

    col_conf_a, col_conf_b = st.columns(2)
    with col_conf_a:
        st.markdown("#### Configuration A")
        strat_a = st.selectbox("Strategy A", ["fixed_size", "structure_aware", "semantic"], index=0, key="strat_a")
        method_a = st.selectbox("Method A", ["hybrid", "dense", "sparse"], index=0, key="method_a")
    with col_conf_b:
        st.markdown("#### Configuration B")
        strat_b = st.selectbox("Strategy B", ["fixed_size", "structure_aware", "semantic"], index=1, key="strat_b")
        method_b = st.selectbox("Method B", ["hybrid", "dense", "sparse"], index=0, key="method_b")

    if st.button("Run Comparison", type="primary") and compare_question:
        col_res_a, col_res_b = st.columns(2)

        payload_a = {
            "query": compare_question,
            "strategy": strat_a,
            "method": method_a,
            "rerank": True,
            "verify_citations": verify_citations,
        }
        payload_b = {
            "query": compare_question,
            "strategy": strat_b,
            "method": method_b,
            "rerank": True,
            "verify_citations": verify_citations,
        }

        with st.spinner("Generating side-by-side answers..."):
            res_a, res_b = None, None
            try:
                # Synchronous requests for simplicity
                resp_a = httpx.post(f"{API_BASE_URL}/ask", json=payload_a, timeout=120.0)
                resp_b = httpx.post(f"{API_BASE_URL}/ask", json=payload_b, timeout=120.0)
                if resp_a.status_code == 200:
                    res_a = resp_a.json()
                if resp_b.status_code == 200:
                    res_b = resp_b.json()
            except Exception as e:
                st.error(f"Error fetching comparison: {e}")

        # Render Left Column
        with col_res_a:
            st.markdown(f"### Answer (Strategy: **{strat_a}**)")
            if res_a:
                st.write(res_a["generation"]["answer"])
                st.markdown(f"**Grounded:** {res_a['generation']['is_grounded']}")
                if verify_citations and res_a["generation"]["is_grounded"]:
                    st.markdown(f"**Citation Support Ratio:** {res_a['support_ratio']*100:.1f}%")
                st.markdown("---")
                st.markdown("**Cited Chunks:**")
                for c in res_a["generation"]["cited_chunks"]:
                    st.caption(f"[{c['index']}] {c['source_file']} — {c['section_heading']}")
            else:
                st.warning("Failed to load Configuration A.")

        # Render Right Column
        with col_res_b:
            st.markdown(f"### Answer (Strategy: **{strat_b}**)")
            if res_b:
                st.write(res_b["generation"]["answer"])
                st.markdown(f"**Grounded:** {res_b['generation']['is_grounded']}")
                if verify_citations and res_b["generation"]["is_grounded"]:
                    st.markdown(f"**Citation Support Ratio:** {res_b['support_ratio']*100:.1f}%")
                st.markdown("---")
                st.markdown("**Cited Chunks:**")
                for c in res_b["generation"]["cited_chunks"]:
                    st.caption(f"[{c['index']}] {c['source_file']} — {c['section_heading']}")
            else:
                st.warning("Failed to load Configuration B.")

# ==========================================
# 3. Evaluation Tab
# ==========================================
with tab_eval:
    st.subheader("Benchmark Comparison Reports")
    st.markdown("Displays the consolidated benchmark reports for each strategy based on 52 golden queries.")

    if st.button("Refresh Evaluation Reports"):
        st.rerun()

    try:
        compare_resp = httpx.get(f"{API_BASE_URL}/eval/compare", timeout=10.0).json()
        comparison = compare_resp.get("comparison", {})
        reports_summary = compare_resp.get("reports", {})

        if not reports_summary:
            st.info("No evaluation reports found. Run evaluation from the CLI or Index Status tab to generate reports.")
        else:
            # 1. Overall Metrics Chart
            st.markdown("### Overall Metrics Comparison")
            metrics_list = ["retrieval_recall", "retrieval_precision", "citation_accuracy", "faithfulness", "correctness"]
            
            chart_data = {
                "Metric": [],
                "Strategy": [],
                "Score": []
            }
            for m in metrics_list:
                for strat in ["fixed_size", "structure_aware", "semantic"]:
                    score = comparison.get("metrics", {}).get(m, {}).get(strat)
                    if score is not None:
                        chart_data["Metric"].append(m)
                        chart_data["Strategy"].append(strat)
                        chart_data["Score"].append(score)

            df_chart = pd.DataFrame(chart_data)
            
            # Pivot table for tabular display
            df_pivot = df_chart.pivot(index="Metric", columns="Strategy", values="Score")
            st.dataframe(df_pivot.style.format("{:.4f}"))

            # Grouped bar chart
            st.markdown("#### Visual Score Comparison")
            st.bar_chart(
                data=df_chart,
                x="Metric",
                y="Score",
                color="Strategy",
                stack=False
            )

            # 2. Category breakdowns
            st.markdown("### Category-Specific Performance Breakdown")
            cat_list = ["lookup", "multi_hop", "ambiguous", "unanswerable"]
            selected_cat = st.selectbox("Select Query Category to inspect:", cat_list)

            cat_chart_data = {
                "Metric": [],
                "Strategy": [],
                "Score": []
            }
            for m in metrics_list:
                for strat in ["fixed_size", "structure_aware", "semantic"]:
                    score = comparison.get("categories", {}).get(selected_cat, {}).get(m, {}).get(strat)
                    if score is not None:
                        cat_chart_data["Metric"].append(m)
                        cat_chart_data["Strategy"].append(strat)
                        cat_chart_data["Score"].append(score)

            df_cat_chart = pd.DataFrame(cat_chart_data)
            df_cat_pivot = df_cat_chart.pivot(index="Metric", columns="Strategy", values="Score")
            st.dataframe(df_cat_pivot.style.format("{:.4f}"))
            st.bar_chart(
                data=df_cat_chart,
                x="Metric",
                y="Score",
                color="Strategy",
                stack=False
            )

            # List of active reports
            st.markdown("### Raw Report Details")
            for strat, meta in reports_summary.items():
                st.text(f"Strategy: {strat.upper()} | Report ID: {meta['id']} | Timestamp: {meta['timestamp']}")

    except Exception as e:
        st.error(f"Failed to fetch comparison reports: {e}")

# ==========================================
# 4. Index Status Tab
# ==========================================
with tab_status:
    st.subheader("Database & Document Ingestion Status")

    try:
        stats_resp = httpx.get(f"{API_BASE_URL}/documents", timeout=5.0).json()
        if stats_resp.get("status") == "success":
            c_indexes = stats_resp["chromadb_indexes"]
            b_indexes = stats_resp["bm25_indexes"]

            # Display Status Table
            index_rows = []
            for strat in ["fixed_size", "structure_aware", "semantic"]:
                c_info = c_indexes.get(strat, {})
                b_info = b_indexes.get(strat, {})
                index_rows.append({
                    "Strategy": strat,
                    "ChromaDB Status": "✅ Ready" if c_info.get("chunk_count", 0) > 0 else "❌ Empty",
                    "ChromaDB Chunk Count": c_info.get("chunk_count", 0),
                    "BM25 Index Status": "✅ Ready" if b_info.get("exists") else "❌ Missing",
                })
            
            st.table(pd.DataFrame(index_rows))

            # Re-index Panel
            st.markdown("### Run Document Ingestion Pipeline")
            st.warning(
                "⚠️ Running full document ingestion will trigger the Document Loader, Chunker, Embedding, and Indexer pipelines. "
                "This will erase existing database collections and re-embed all documents, which can consume a significant amount of LLM quota."
            )
            
            if st.button("Trigger Full Ingestion Pipeline", type="secondary"):
                with st.spinner("Running ingestion pipeline... (This might take a couple of minutes)"):
                    try:
                        ingest_resp = httpx.post(f"{API_BASE_URL}/ingest", timeout=300.0).json()
                        if ingest_resp.get("status") == "success":
                            st.success("Ingestion pipeline completed successfully!")
                            st.json(ingest_resp["summary"])
                        else:
                            st.error(f"Ingestion pipeline failed: {ingest_resp}")
                    except Exception as ie:
                        st.error(f"Error triggering ingestion: {ie}")
        else:
            st.error("Failed to load document stats.")
    except Exception as e:
        st.error(f"Failed to connect to stats endpoint: {e}")
