"use client"
import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { RecommendFeed } from '../../../components/RecommendFeed';
import Link from 'next/link';

export default function RecommendPage() {
  const { id } = useParams();
  const router = useRouter();
  const [recs, setRecs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!id) return;
    fetch(`http://localhost:8000/users/${id}/recommend?k=12`)
      .then(res => {
        if (!res.ok) throw new Error('Could not fetch recommendations. Ensure the model has been trained and semantic embeddings exist.');
        return res.json();
      })
      .then(data => {
        setRecs(data.recommendations || []);
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

  if (error) {
    return (
      <div className="min-h-screen p-8 flex flex-col justify-center items-center">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-4">Error</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6 text-center max-w-md">{error}</p>
        <button onClick={() => router.push(`/user/${id}`)} className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 transition-colors text-white rounded-md">
          Back to Profile
        </button>
      </div>
    );
  }

  return (
    <div className="p-4 md:p-8 bg-transparent">
      <div className="max-w-6xl mx-auto">
        <div className="mb-6 flex justify-between items-center">
          <Link href={`/user/${id}`} className="text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300 flex items-center font-semibold">
            &larr; Back to Profile
          </Link>
          <div className="text-sm text-gray-500 dark:text-gray-400">
            Denoised Recommendations
          </div>
        </div>

        <header className="mb-8">
          <h1 className="text-3xl font-extrabold text-gray-900 dark:text-gray-100 mb-2">Recommended for User {(id as string).replace('user', '#')}</h1>
          <p className="text-gray-600 dark:text-gray-400">
            These recommendations are generated after <span className="font-semibold text-emerald-600 dark:text-emerald-500">ECHO</span> has filtered out behavioral noise from the user's history, focusing only on their authentic interests.
          </p>
        </header>

        <RecommendFeed recommendations={recs} />
      </div>
    </div>
  );
}
