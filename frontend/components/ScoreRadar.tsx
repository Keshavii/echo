"use client"
import React from 'react';
import { PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer, Tooltip } from "recharts"

export function ScoreRadar({ c1, c2, c3 }: { c1: number, c2: number, c3: number }) {
  const data = [
    { subject: 'Semantic History', A: c1 },
    { subject: 'Collab History', A: c2 },
    { subject: 'Cross-Modal', A: c3 },
  ];

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
          <PolarGrid />
          <PolarAngleAxis dataKey="subject" tick={{ fill: '#666', fontSize: 12 }} />
          <PolarRadiusAxis angle={30} domain={[-1, 1]} tick={{ fontSize: 10 }} />
          <Radar name="Score" dataKey="A" stroke="#4f46e5" fill="#4f46e5" fillOpacity={0.5} />
          <Tooltip />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  )
}
