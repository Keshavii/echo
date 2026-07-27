import React from 'react';

export function ItemCard({ title, isKept, score, explanation, c1, c2, c3 }: any) {
  return (
    <div className={`p-4 border rounded-lg shadow-sm transition-all ${isKept ? 'border-emerald-500 bg-emerald-50/20 dark:bg-emerald-900/10' : 'border-rose-500 bg-rose-50/20 dark:bg-rose-900/10'}`}>
      <div className="flex justify-between items-start mb-2">
        <h3 className="text-md font-semibold text-gray-800 dark:text-gray-200 line-clamp-2 pr-4">{title}</h3>
        <span className={`px-2 py-1 text-xs font-bold rounded-full whitespace-nowrap ${isKept ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300' : 'bg-rose-100 text-rose-800 dark:bg-rose-900/40 dark:text-rose-300'}`}>
          {isKept ? 'AUTHENTIC' : 'NOISE'}
        </span>
      </div>
      
      {explanation && (
        <div className="text-sm text-gray-600 dark:text-gray-400 italic mb-3">
          {explanation}
        </div>
      )}

      {(c1 !== undefined && c2 !== undefined && c3 !== undefined) && (
        <div className="flex gap-4 text-xs mt-3 pt-3 border-t border-gray-200 dark:border-gray-800">
          <div className="flex flex-col">
            <span className="text-gray-400 dark:text-gray-500">Semantic</span>
            <span className={c1 > 0 ? "text-emerald-600" : "text-rose-600"}>{c1.toFixed(2)}</span>
          </div>
          <div className="flex flex-col">
            <span className="text-gray-400 dark:text-gray-500">Collaborative</span>
            <span className={c2 > 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}>{c2.toFixed(2)}</span>
          </div>
          <div className="flex flex-col">
            <span className="text-gray-400 dark:text-gray-500">Cross-Modal</span>
            <span className={c3 > 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}>{c3.toFixed(2)}</span>
          </div>
          {score !== undefined && (
            <div className="flex flex-col ml-auto">
              <span className="text-gray-400 dark:text-gray-500 font-bold">Total Score</span>
              <span className={score > 0 ? "text-emerald-600 dark:text-emerald-400 font-bold" : "text-rose-600 dark:text-rose-400 font-bold"}>{score.toFixed(2)}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
