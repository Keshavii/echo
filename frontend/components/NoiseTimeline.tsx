import React from 'react';
import { ItemCard } from './ItemCard';

export function NoiseTimeline({ history }: { history: any[] }) {
  if (!history || history.length === 0) {
    return <div className="p-4 text-gray-500 dark:text-gray-400">No history available for this user.</div>;
  }

  return (
    <div className="relative pl-8 border-l-2 border-gray-200 dark:border-gray-800 py-4 space-y-6">
      {history.map((item, index) => (
        <div key={index} className="relative">
          {/* Timeline Dot */}
          <div className={`absolute -left-[41px] top-4 w-5 h-5 rounded-full border-4 border-white dark:border-gray-900 ${item.kept ? 'bg-emerald-500' : 'bg-rose-500'}`} />
          
          <ItemCard 
            title={item.title}
            isKept={item.kept}
            score={item.score}
            explanation={item.explanation}
            c1={item.c1}
            c2={item.c2}
            c3={item.c3}
          />
        </div>
      ))}
    </div>
  );
}
