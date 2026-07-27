"use client"
import { useEffect, useState } from 'react';
import Link from 'next/link';
import { MetricsDashboard } from '../../components/MetricsDashboard';

export default function AnalyticsPage() {
  const [metrics, setMetrics] = useState<any>(null);
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch('http://localhost:8000/metrics').then(res => res.json()),
      fetch('http://localhost:8000/stats').then(res => res.json())
    ])
      .then(([metricsData, statsData]) => {
        setMetrics(metricsData);
        setStats(statsData);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to fetch analytics", err);
        setLoading(false);
      });
  }, []);

  return (
    <div className="p-4 md:p-8 bg-transparent">
      <div className="max-w-6xl mx-auto">
        <div className="mb-6">
          <Link href="/" className="text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300 flex items-center font-semibold w-max">
            &larr; Back to Home
          </Link>
        </div>

        <header className="mb-10 text-center">
          <h1 className="text-4xl font-extrabold text-gray-900 dark:text-gray-100 mb-4">ECHO System Analytics</h1>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            Performance evaluation of the IADSR+ model against standard baselines, measured on the Amazon Beauty dataset.
          </p>
        </header>

        {loading ? (
          <div className="flex justify-center py-20">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 dark:border-indigo-400"></div>
          </div>
        ) : (
          <>
            {stats && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
                <div className="bg-white dark:bg-gray-900 p-5 rounded-xl border border-gray-200 dark:border-gray-800 text-center shadow-sm">
                  <div className="text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">Dataset</div>
                  <div className="text-xl font-bold text-gray-900 dark:text-gray-100">{stats.dataset}</div>
                </div>
                <div className="bg-white dark:bg-gray-900 p-5 rounded-xl border border-gray-200 dark:border-gray-800 text-center shadow-sm">
                  <div className="text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">Users Analyzed</div>
                  <div className="text-xl font-bold text-gray-900 dark:text-gray-100">{stats.total_users.toLocaleString()}</div>
                </div>
                <div className="bg-white dark:bg-gray-900 p-5 rounded-xl border border-gray-200 dark:border-gray-800 text-center shadow-sm">
                  <div className="text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">Total Items</div>
                  <div className="text-xl font-bold text-gray-900 dark:text-gray-100">{stats.total_items.toLocaleString()}</div>
                </div>
                <div className="bg-white dark:bg-gray-900 p-5 rounded-xl border border-gray-200 dark:border-gray-800 text-center shadow-sm">
                  <div className="text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">Avg Sequence</div>
                  <div className="text-xl font-bold text-gray-900 dark:text-gray-100">{stats.avg_seq_len} items</div>
                </div>
              </div>
            )}

            <MetricsDashboard metrics={metrics} />
            
            {stats && stats.novel_contrib && (
              <div className="mt-12 bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-100 dark:border-indigo-800/50 p-6 rounded-2xl">
                <h3 className="text-lg font-bold text-indigo-900 dark:text-indigo-100 mb-3">Model Architecture Improvements (IADSR+)</h3>
                <ul className="list-disc pl-5 space-y-2 text-indigo-800 dark:text-indigo-300">
                  {stats.novel_contrib.map((contrib: string, i: number) => (
                    <li key={i}>{contrib}</li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
