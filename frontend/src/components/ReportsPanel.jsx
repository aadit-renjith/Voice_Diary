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
  surprised: "😲"
};

const ReportsPanel = () => {

  const [weekly,setWeekly] = useState(null);
  const [monthly,setMonthly] = useState(null);

  useEffect(() => {
    fetchReports();
  }, []);

  const fetchReports = async () => {

    try {

      const [w,m] = await Promise.all([
        axios.get(`${API}/analytics/weekly`),
        axios.get(`${API}/analytics/monthly`)
      ]);

      setWeekly(w.data);
      setMonthly(m.data);

    } catch(err) {
      console.error(err);
    }

  };

  const renderDist = (dist) => {

    if(!dist) return null;

    return Object.entries(dist).map(([emotion,count]) => (

      <div key={emotion} className="dist-row">

        <span className="dist-label">
          {emoji[emotion]} {emotion}
        </span>

        <div className="dist-bar">
          <div
            className="dist-fill"
            style={{ width: `${count * 20}px` }}
          />
        </div>

        <span className="dist-count">
          {count}
        </span>

      </div>

    ));

  };

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
                  {emoji[weekly.dominant_emotion]} {weekly.dominant_emotion || "N/A"}
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