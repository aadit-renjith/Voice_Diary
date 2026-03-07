import React, { useEffect, useState } from "react";
import axios from "axios";
import "./ReportsPanel.css";

const API = "http://127.0.0.1:8000";

const emoji = {
  happy: "🤩",
  sad: "😢",
  angry: "😠",
  fearful: "😰",
  neutral: "😐",
  surprised: "😲",
  calm: "😌",
  disgust: "🤢"
};

const ReportsPanel = ({ refreshKey }) => {

  const [weekly, setWeekly] = useState(null);
  const [monthly, setMonthly] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchReports();
  }, [refreshKey]);

  const fetchReports = async () => {
    setLoading(true);
    setError(null);
    try {
      const [w, m] = await Promise.all([
        axios.get(`${API}/analytics/weekly`),
        axios.get(`${API}/analytics/monthly`)
      ]);
      setWeekly(w.data);
      setMonthly(m.data);
    } catch (err) {
      console.error("Failed to fetch reports:", err);
      setError("Could not load reports. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const renderDist = (dist) => {
    if (!dist || Object.keys(dist).length === 0) {
      return <p className="dist-empty">No emotion data recorded yet.</p>;
    }

    const entries = Object.entries(dist);
    const maxCount = Math.max(...entries.map(([, count]) => count), 1);

    return entries.map(([emotion, count]) => (
      <div key={emotion} className="dist-row">
        <span className="dist-label">
          {emoji[emotion] || "💭"} {emotion}
        </span>
        <div className="dist-bar">
          <div
            className="dist-fill"
            style={{ width: `${(count / maxCount) * 100}%` }}
          />
        </div>
        <span className="dist-count">{count}</span>
      </div>
    ));
  };

  if (loading) {
    return (
      <div className="reports-container">
        <div className="report-card">
          <h3>Weekly Report</h3>
          <p className="dist-empty">Loading...</p>
        </div>
        <div className="report-card">
          <h3>Monthly Report</h3>
          <p className="dist-empty">Loading...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="reports-container">
        <div className="report-card report-card-error">
          <h3>Reports</h3>
          <p className="dist-empty">{error}</p>
          <button className="retry-btn" onClick={fetchReports}>Retry</button>
        </div>
      </div>
    );
  }

  return (
    <div className="reports-container">

      {/* Weekly */}
      <div className="report-card">
        <h3>Weekly Report</h3>
        {weekly && (
          <>
            <div className="report-metrics">
              <div className="metric">
                <span>Total Entries</span>
                <b>{weekly.total_entries}</b>
              </div>
              <div className="metric">
                <span>Dominant Emotion</span>
                <b>
                  {weekly.dominant_emotion
                    ? `${emoji[weekly.dominant_emotion] || "💭"} ${weekly.dominant_emotion}`
                    : "N/A"}
                </b>
              </div>
            </div>
            <div className="dist">
              {renderDist(weekly.emotion_distribution)}
            </div>
          </>
        )}
      </div>

      {/* Monthly */}
      <div className="report-card">
        <h3>Monthly Report</h3>
        {monthly && (
          <>
            <div className="metric">
              <span>Total Entries</span>
              <b>{monthly.total_entries}</b>
            </div>
            <div className="dist">
              {renderDist(monthly.emotion_distribution)}
            </div>
          </>
        )}
      </div>

    </div>
  );

};

export default ReportsPanel;