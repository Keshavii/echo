"use client"
import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export function MetricsDashboard({ metrics }: { metrics: any }) {
  if (!metrics || !metrics.IADSR_plus_ECHO || !metrics.paper_baselines) {
    return (
      <div className="p-8 text-center text-gray-500 bg-white rounded-xl border border-gray-200">
        <p className="mb-2 text-lg">Evaluation metrics not available yet.</p>
        <p className="text-sm">Run <code className="bg-gray-100 px-1 rounded">python evaluate.py</code> in the <code className="bg-gray-100 px-1 rounded">ml/</code> directory after training.</p>
      </div>
    );
  }

  const { IADSR_plus_ECHO: echo, paper_baselines: baselines } = metrics;
  const original = baselines.IADSR_original_paper;
  const gru = baselines.GRU4Rec_base;

  const data = [
    {
      name: 'HR@5',
      'GRU4Rec (Baseline)': gru["5"]?.HR || 0.0153,
      'IADSR (Original)': original["5"]?.HR || 0.0300,
      'ECHO (IADSR+)': echo["5"]?.HR || 0,
    },
    {
      name: 'HR@10',
      'GRU4Rec (Baseline)': gru["10"]?.HR || 0.0246,
      'IADSR (Original)': original["10"]?.HR || 0.0396,
      'ECHO (IADSR+)': echo["10"]?.HR || 0,
    },
    {
      name: 'HR@20',
      'GRU4Rec (Baseline)': gru["20"]?.HR || 0.0390,
      'IADSR (Original)': original["20"]?.HR || 0.0486,
      'ECHO (IADSR+)': echo["20"]?.HR || 0,
    },
  ];

  const ndcgData = [
    {
      name: 'NDCG@5',
      'GRU4Rec (Baseline)': gru["5"]?.NDCG || 0.0087,
      'IADSR (Original)': original["5"]?.NDCG || 0.0196,
      'ECHO (IADSR+)': echo["5"]?.NDCG || 0,
    },
    {
      name: 'NDCG@10',
      'GRU4Rec (Baseline)': gru["10"]?.NDCG || 0.0117,
      'IADSR (Original)': original["10"]?.NDCG || 0.0259,
      'ECHO (IADSR+)': echo["10"]?.NDCG || 0,
    },
    {
      name: 'NDCG@20',
      'GRU4Rec (Baseline)': gru["20"]?.NDCG || 0.0143,
      'IADSR (Original)': original["20"]?.NDCG || 0.0321,
      'ECHO (IADSR+)': echo["20"]?.NDCG || 0,
    },
  ];

  return (
    <div className="space-y-10">
      <div className="bg-white dark:bg-gray-900 p-6 rounded-2xl border border-gray-200 dark:border-gray-800 shadow-sm">
        <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-6 text-center">Hit Ratio (HR@K)</h3>
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" strokeOpacity={0.2} />
              <XAxis dataKey="name" tick={{fill: '#9CA3AF'}} axisLine={false} tickLine={false} />
              <YAxis tick={{fill: '#9CA3AF'}} axisLine={false} tickLine={false} />
              <Tooltip cursor={{fill: 'rgba(156, 163, 175, 0.1)'}} contentStyle={{borderRadius: '8px', border: 'none', backgroundColor: '#1F2937', color: '#F9FAFB', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}} />
              <Legend wrapperStyle={{paddingTop: '20px'}} />
              <Bar dataKey="GRU4Rec (Baseline)" fill="#94A3B8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="IADSR (Original)" fill="#818CF8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="ECHO (IADSR+)" fill="#4F46E5" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-900 p-6 rounded-2xl border border-gray-200 dark:border-gray-800 shadow-sm">
        <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-6 text-center">Normalized Discounted Cumulative Gain (NDCG@K)</h3>
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={ndcgData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" strokeOpacity={0.2} />
              <XAxis dataKey="name" tick={{fill: '#9CA3AF'}} axisLine={false} tickLine={false} />
              <YAxis tick={{fill: '#9CA3AF'}} axisLine={false} tickLine={false} />
              <Tooltip cursor={{fill: 'rgba(156, 163, 175, 0.1)'}} contentStyle={{borderRadius: '8px', border: 'none', backgroundColor: '#1F2937', color: '#F9FAFB', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}} />
              <Legend wrapperStyle={{paddingTop: '20px'}} />
              <Bar dataKey="GRU4Rec (Baseline)" fill="#94A3B8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="IADSR (Original)" fill="#818CF8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="ECHO (IADSR+)" fill="#10B981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
