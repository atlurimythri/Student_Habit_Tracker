import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import datetime

st.set_page_config(page_title="Student Behavior System", layout="wide")

# ---------------------------
# SESSION STATE
# ---------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None
if "session_start" not in st.session_state:
    st.session_state.session_start = None


# ---------------------------
# LOGIN
# ---------------------------
def login():
    st.title("🎓 Student Behavior Analysis System")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.session_state.role = "admin"
            st.session_state.session_start = datetime.datetime.now()

        elif username == "user" and password == "user123":
            st.session_state.logged_in = True
            st.session_state.role = "user"
            st.session_state.session_start = datetime.datetime.now()
        else:
            st.error("Invalid Credentials")


# ---------------------------
# LOGOUT
# ---------------------------
def logout():
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.session_start = None


# ---------------------------
# USER DASHBOARD
# ---------------------------
def user_dashboard():
    df = pd.read_csv("processed_student_data.csv")
    df.columns = df.columns.str.lower()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Menu",
        ["Dashboard", "Log Study Session", "Analysis"]
    )
    st.sidebar.button("Logout", on_click=logout)
    st.title("📊 Student Dashboard")
    # ===============================
    # DASHBOARD PAGE
    # ===============================
    if page == "Dashboard":
        user_id = st.selectbox(
            "Enter Your Student ID",
            sorted(df["student_id"].unique())
        )
        sessions = st.slider(
            "Study Sessions Per Day",
            1,
            5,
            2
        )
        if "cluster" not in df.columns:
            st.warning("Admin must run clustering first.")
            return
        if st.button("Generate Recommendation"):
            user_data = df[df["student_id"] == int(user_id)]
  
            if user_data.empty:
                st.error("Student not found.")
                return

            user_data = user_data.iloc[0]
            cluster = user_data["cluster"]

            if cluster == 0:
                routine = "Morning Deep Study"
            elif cluster == 1:
                routine = "Short Burst Sessions"
            elif cluster == 2:
                routine = "Evening Study Routine"
            else:
                routine = "Distraction Controlled Study"

            st.success(f"Recommended Study Style: {routine}")

            # -----------------------
            # PERFORMANCE PROGRESS
            # -----------------------
            st.subheader("Performance Progress")

            days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

            study = np.random.randint(1,4,7)
            quiz = np.random.randint(60,90,7)

            fig, ax = plt.subplots()

            ax.plot(days, study, label="Study Time")
            ax.plot(days, quiz, label="Quiz Score")

            ax.set_ylabel("Progress")
            ax.legend()

            st.pyplot(fig)

            # -----------------------
            # TODAY ROUTINE
            # -----------------------
            st.subheader("Today's Recommended Routine")

            st.info(f"""
            8:00 AM — Mathematics Study ({sessions*45} mins)

            10:00 AM — Practice Problems

            4:00 PM — Revision Session

            Evening — Quick Quiz
            """)

    # ===============================
    # LOG STUDY SESSION
    # ===============================
    elif page == "Log Study Session":

        st.subheader("Log Study Session")

        col1, col2 = st.columns(2)

        with col1:

            date = st.date_input("Date")

            study_time = st.selectbox(
                "Study Time",
                ["Morning", "Afternoon", "Evening", "Night"]
            )

            distractions = st.selectbox(
                "Distractions",
                ["None", "Phone", "Social Media", "Noise"]
            )

        with col2:

            duration = st.number_input(
                "Study Duration (minutes)",
                min_value=10,
                max_value=240,
                step=10
            )

            subject = st.selectbox(
                "Subject",
                ["Mathematics","Science","Programming","English"]
            )

            quiz_score = st.slider(
                "Quiz Score (%)",
                0,
                100,
                70
            )

        if st.button("Save Log"):

            new_log = pd.DataFrame([{
                "date": date,
                "study_time": study_time,
                "duration": duration,
                "subject": subject,
                "quiz_score": quiz_score,
                "distractions": distractions
            }])

            try:
                old = pd.read_csv("study_logs.csv")
                updated = pd.concat([old, new_log], ignore_index=True)
            except:
                updated = new_log

            updated.to_csv("study_logs.csv", index=False)

            st.success("Study session saved successfully!")

            st.rerun()

    # ===============================
    # ANALYSIS PAGE
    # ===============================
    elif page == "Analysis":

        st.subheader("📊 Student Behavior Analysis")

        try:
            log_df = pd.read_csv("study_logs.csv")
        except:
            st.warning("No study sessions logged yet.")
            return

        st.markdown("### Recent Study Logs")
        st.dataframe(log_df.tail(10))

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Study Duration Distribution")

            fig, ax = plt.subplots()
            ax.hist(log_df["duration"])
            ax.set_xlabel("Minutes")
            ax.set_ylabel("Sessions")

            st.pyplot(fig)

        with col2:

            st.subheader("Quiz Score Trend")

            fig2, ax2 = plt.subplots()
            ax2.plot(log_df["quiz_score"])

            ax2.set_ylabel("Score")
            ax2.set_xlabel("Session")

            st.pyplot(fig2)

        # -----------------------
        # Subject Analysis
        # -----------------------
        st.subheader("Subject Study Frequency")

        subject_counts = log_df["subject"].value_counts()

        fig3, ax3 = plt.subplots()
        ax3.bar(subject_counts.index, subject_counts.values)
        ax3.set_ylabel("Sessions")
        ax3.set_title("Subjects Studied")

        st.pyplot(fig3)
# ---------------------------
# ADMIN DASHBOARD
# ---------------------------
def admin_dashboard():

    st.title("🛠 Admin Dashboard")

    st.sidebar.title("📂 Admin Menu")

    # ---------------------------
    # NEW: DATASET DROPDOWN
    # ---------------------------
    dataset_option = st.sidebar.selectbox(
        "Select Dataset",
        ["Default Dataset", "Upload New CSV"]
    )

    if dataset_option == "Upload New CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV File",
            type=["csv"]
        )
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("CSV Uploaded Successfully!")
        else:
            df = pd.read_csv("processed_student_data.csv")
    else:
        df = pd.read_csv("processed_student_data.csv")

    # ---------------------------
    # EXISTING MENU
    # ---------------------------
    page = st.sidebar.radio(
        "Select Section",
        ["Milestone 1 - EDA", "Milestone 2 - Clustering"]
    )

    st.sidebar.button("🚪 Logout", on_click=logout)

    # =============================
    # MILESTONE 1 (UNCHANGED)
    # =============================
    if page == "Milestone 1 - EDA": 

        st.markdown("## 📊 Milestone 1: Data Preprocessing & EDA")

        st.markdown("### 🔎 Filter Data")

        f1, f2, f3 = st.columns(3)

        with f1:
            study_range = st.slider("Study Hours",
                                    float(df.study_hours.min()),
                                    float(df.study_hours.max()),
                                    (float(df.study_hours.min()),
                                     float(df.study_hours.max())))

        with f2:
            score_range = st.slider("Post Test Score",
                                    float(df.post_test.min()),
                                    float(df.post_test.max()),
                                    (float(df.post_test.min()),
                                     float(df.post_test.max())))

        with f3:
            attention_range = st.slider("Attention Span",
                                        float(df.attention_span.min()),
                                        float(df.attention_span.max()),
                                        (float(df.attention_span.min()),
                                         float(df.attention_span.max())))

        filtered_df = df[
            (df.study_hours.between(*study_range)) &
            (df.post_test.between(*score_range)) &
            (df.attention_span.between(*attention_range))
        ]

        if len(filtered_df) == 0:
            st.warning("No data available.")
            return

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            sns.scatterplot(data=filtered_df,
                            x="study_hours",
                            y="post_test",
                            ax=ax)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            sns.heatmap(filtered_df.corr(),
                        cmap="coolwarm",
                        annot=True,
                        ax=ax)
            st.pyplot(fig)

    # =============================
    # MILESTONE 2 (UNCHANGED)
    # =============================
    elif page == "Milestone 2 - Clustering":

        st.markdown("""
        <div style='background-color:#2E3B8C;padding:20px;border-radius:10px'>
            <h2 style='color:white'>Milestone 2: Clustering & Pattern Detection</h2>
            <p style='color:white'>
            Identify student behavior types using KMeans clustering.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("## 📊 Student Behavior Clustering Dashboard")

        p1, p2, p3 = st.columns(3)

        with p1:
            n_clusters = st.slider("Number of Clusters", 2, 6, 4)

        with p2:
            st.button("🔄 Retrain Model")

        with p3:
            st.button("📥 Export Data")

        features = df[["study_hours", "pre_test",
                       "post_test", "attention_span"]]

        scaler = StandardScaler()
        X = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df["Cluster"] = kmeans.fit_predict(X)

        # ✅ SAVE CLUSTERED DATA
        df.to_csv("processed_student_data.csv", index=False)

        cluster_means = df.groupby("Cluster").mean(numeric_only=True)
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Student Behavior Clusters")

            fig1, ax1 = plt.subplots()
            sns.scatterplot(data=df,
                            x="study_hours",
                            y="post_test",
                            hue="Cluster",
                            palette="Set2",
                            ax=ax1)
            ax1.set_xlim(left=0)
            ax1.set_ylim(bottom=0)
            ax1.grid(True)
            st.pyplot(fig1)

        with col2:
            st.subheader("Cluster Distribution")

            cluster_counts = df["Cluster"].value_counts()

            for cluster, count in cluster_counts.items():
                percent = round((count / len(df)) * 100, 1)
                st.metric(f"Cluster {cluster}",
                          f"{percent}%")

        st.markdown("---")

        col3, col4 = st.columns([2, 1])

        with col3:
            st.subheader("Cluster Characteristics (Radar Chart)")

            selected_cluster = st.selectbox(
                "Select Cluster",
                cluster_means.index
            )

            values = cluster_means.loc[selected_cluster][
                ["study_hours",
                 "post_test",
                 "attention_span",
                 "pre_test"]
            ].values

            categories = ["Study", "Post Test",
                          "Attention", "Pre Test"]

            angles = np.linspace(0, 2*np.pi,
                                 len(categories),
                                 endpoint=False)

            values = np.concatenate((values, [values[0]]))
            angles = np.concatenate((angles, [angles[0]]))

            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111, polar=True)

            ax2.plot(angles, values)
            ax2.fill(angles, values, alpha=0.3)
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(categories)

            st.pyplot(fig2)

        with col4:
            st.subheader("Cluster Profile")

            profile = cluster_means.loc[selected_cluster]

            st.metric("Avg Study Hours",
                      round(profile["study_hours"], 2))
            st.metric("Avg Post Test",
                      round(profile["post_test"], 2))
            st.metric("Avg Attention",
                      round(profile["attention_span"], 2))
            st.metric("Avg Pre Test",
                      round(profile["pre_test"], 2))


# ---------------------------
# MAIN
# ---------------------------
if not st.session_state.logged_in:
    login()
else:
    if st.session_state.role == "admin":
        admin_dashboard()
    else:    
        user_dashboard()
