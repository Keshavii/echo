"use client"
import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { NoiseTimeline } from '../../../components/NoiseTimeline';
import { ScoreRadar } from '../../../components/ScoreRadar';
import Link from 'next/link';

export default function UserProfile() {
  const { id } = useParams();
  const router = useRouter();
  const [profile, setProfile] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!id) return;
    fetch(`http://localhost:8000/users/${id}/history`)
      .then(res => {
        if (!res.ok) throw new Error('User not found or has insufficient data');
        return res.json();
      })
      .then(data => {
        setProfile(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [id]);

  if (loading) {
    return (
      <div className="min-h-screen p-8 flex justify-center items-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 dark:border-indigo-400"></div>
      </div>
    );
  }

  if (error || !profile) {
    return (
      <div className="min-h-screen p-8 flex flex-col justify-center items-center">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-4">Error</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">{error}</p>
        <button onClick={() => router.push('/')} className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-md">
          Go Back
        </button>
      </div>
    );
  }

  // Get the most recent interaction with scores for the radar chart
  const itemsWithScores = profile.history.filter((h: any) => h.c1 !== null).reverse();
  const latestItem = itemsWithScores.length > 0 ? itemsWithScores[0] : null;

  return (
    <div className="p-4 md:p-8 bg-transparent">
      <div className="max-w-6xl mx-auto">
        <div className="mb-6 flex justify-between items-center">
          <Link href="/" className="text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300 flex items-center font-semibold bg-white dark:bg-gray-900 px-3 py-1.5 rounded-md shadow-sm border border-gray-200 dark:border-gray-800">
            &larr; Back to Users
          </Link>
          
          <div className="flex items-center gap-4">
            <div className="text-sm font-medium text-gray-500 dark:text-gray-400 bg-white dark:bg-gray-900 px-3 py-1.5 rounded-md shadow-sm border border-gray-200 dark:border-gray-800">
              Cross-Modal Sim: <span className="text-gray-900 dark:text-gray-100 font-bold">{profile.cross_modal_sim}</span>
            </div>
            <Link href={`/recommend/${id}`} className="px-4 py-1.5 bg-indigo-600 hover:bg-indigo-700 dark:bg-indigo-500 dark:hover:bg-indigo-600 text-white text-sm font-medium rounded-md shadow-sm transition-colors flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m5 12 7-7 7 7"/><path d="M12 19V5"/></svg>
              View Recommendations
            </Link>
          </div>
        </div>

        <header className="mb-8 p-6 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-2xl shadow-sm">
          <h1 className="text-3xl font-extrabold text-gray-900 dark:text-gray-100 mb-2">User {profile.user_id.replace('user', '#')}</h1>
          <div className="flex gap-4 mt-4">
            <div className="px-4 py-2 bg-gray-100 dark:bg-gray-800 rounded-lg">
              <span className="block text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wide">Total Items</span>
              <span className="text-lg font-bold text-gray-900 dark:text-gray-100">{profile.history.length}</span>
            </div>
            <div className="px-4 py-2 bg-emerald-50 dark:bg-emerald-900/30 rounded-lg border border-emerald-100 dark:border-emerald-800/50">
              <span className="block text-xs text-emerald-600 dark:text-emerald-400 uppercase tracking-wide">Authentic Interests</span>
              <span className="text-lg font-bold text-emerald-700 dark:text-emerald-300">{profile.keep_count}</span>
            </div>
            <div className="px-4 py-2 bg-rose-50 dark:bg-rose-900/30 rounded-lg border border-rose-100 dark:border-rose-800/50">
              <span className="block text-xs text-rose-600 dark:text-rose-400 uppercase tracking-wide">Noise Removed</span>
              <span className="text-lg font-bold text-rose-700 dark:text-rose-300">{profile.noise_count}</span>
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-6">Interaction Timeline</h2>
            <div className="bg-white dark:bg-gray-900 p-6 rounded-2xl border border-gray-200 dark:border-gray-800 shadow-sm">
              <NoiseTimeline history={profile.history} />
            </div>
          </div>
          
          <div className="lg:col-span-1">
            <div className="sticky top-24">
              <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-6">Latest Interaction Analysis</h2>
              <div className="bg-white dark:bg-gray-900 p-6 rounded-2xl border border-gray-200 dark:border-gray-800 shadow-sm">
                {latestItem ? (
                  <>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-4 text-center">
                      Analysis for: <span className="font-semibold text-gray-900 dark:text-gray-100">{latestItem.title}</span>
                    </p>
                    <ScoreRadar c1={latestItem.c1} c2={latestItem.c2} c3={latestItem.c3} />
                    <div className="mt-4 p-4 rounded-lg bg-gray-50 dark:bg-gray-800/50 text-sm italic text-gray-700 dark:text-gray-300 text-center">
                      {latestItem.explanation}
                    </div>
                  </>
                ) : (
                  <p className="text-gray-500 dark:text-gray-400 text-center py-10">No scored interactions yet.</p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
