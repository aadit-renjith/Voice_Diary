import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import './EmotionChart.css';

const EmotionChart = ({ data }) => {
    const moodScore = { happy: 2, surprised: 1, neutral: 0, calm: 0, angry: -1, fearful: -1.5, sad: -2, disgust: -2 };

    const processed = data.map((e, i) => ({
        name: i + 1,
        score: moodScore[e.emotion] || 0,
        emotion: e.emotion,
    })).slice(-15);

    return (
        <div className="chart-inner">
            <p className="mood-title">Mood Trend</p>
            <div className="chart-area">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={processed} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                        <defs>
                            <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.5} />
                                <stop offset="95%" stopColor="#22d3ee" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                        <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} label={{ value: 'Entry', position: 'insideBottomRight', offset: -5, fill: '#64748b', fontSize: 11 }} />
                        <YAxis stroke="#64748b" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} domain={[-2.5, 2.5]} label={{ value: 'Mood Score', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }} />
                        <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#e2e8f0' }} itemStyle={{ color: '#22d3ee' }} />
                        <Area type="monotone" dataKey="score" stroke="#22d3ee" strokeWidth={2.5} fill="url(#grad)" dot={{ r: 4, fill: '#22d3ee', stroke: '#0f172a', strokeWidth: 2 }} activeDot={{ r: 6 }} />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
            <p className="chart-note">Higher score = Positive (Happy), Lower = Negative (Sad/Angry)</p>
        </div>
    );
};

export default EmotionChart;
