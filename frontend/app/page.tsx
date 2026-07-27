"use client"
import { useEffect, useState } from 'react';
import Link from 'next/link';

export default function Home() {
  const [users, setUsers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://localhost:8000/users?limit=20')
      .then(res => res.json())
      .then(data => {
        setUsers(data.users || []);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to fetch users", err);
        setLoading(false);
      });
  }, []);

  return (
    <div className="p-8 bg-transparent">
      <div className="max-w-4xl mx-auto">
        <header className="mb-12 text-center relative">
          <h1 className="text-4xl font-extrabold tracking-tight text-gray-900 dark:text-gray-100 sm:text-5xl mb-4">
            ECHO
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Your clicks lie. Your true interests don't.
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-500 mt-2">
            Select a user below to view their interaction history and how ECHO denoises their digital footprint.
          </p>
        </header>

        {loading ? (
          <div className="text-center py-20">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
            <p className="mt-4 text-gray-600 dark:text-gray-400">Loading users...</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {users.map((user) => (
              <Link key={user.user_id} href={`/user/${user.user_id}`}>
                <div className="block p-6 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl shadow-sm hover:shadow-md hover:border-indigo-500 dark:hover:border-indigo-400 transition-all cursor-pointer">
                  <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-2">{user.display_name}</h2>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    {user.interaction_count} Interactions
                  </p>
                  <div className="mt-4 flex items-center text-indigo-600 dark:text-indigo-400 text-sm font-semibold">
                    View Profile &rarr;
                  </div>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
