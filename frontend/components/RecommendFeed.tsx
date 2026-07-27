import React from 'react';

export function RecommendFeed({ recommendations }: { recommendations: any[] }) {
  if (!recommendations || recommendations.length === 0) {
    return <div className="text-gray-500 dark:text-gray-400 py-10 text-center">No recommendations available.</div>;
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
      {recommendations.map((rec, idx) => (
        <div key={rec.item_id} className="bg-white dark:bg-gray-900 p-5 border border-indigo-100 dark:border-indigo-900/30 rounded-xl shadow-sm hover:shadow-md hover:border-indigo-300 dark:hover:border-indigo-700 transition-all flex flex-col h-full relative overflow-hidden group">
          {idx < 3 && (
            <div className="absolute top-0 right-0 bg-indigo-600 dark:bg-indigo-500 text-white text-[10px] font-bold px-3 py-1 rounded-bl-lg z-10">
              TOP {idx + 1}
            </div>
          )}
          <div className="flex-grow z-0">
            <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2 line-clamp-3 pr-4 group-hover:text-indigo-700 dark:group-hover:text-indigo-400 transition-colors">{rec.title}</h3>
          </div>
          <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-800 flex justify-between items-center">
            <span className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Model Score</span>
            <span className="text-sm font-bold text-indigo-600 dark:text-indigo-400">{(rec.score * 10).toFixed(2)}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
