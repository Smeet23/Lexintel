import React from 'react';

export function Spinner() {
  return (
    <div className="inline-block">
      <div className="h-8 w-8 animate-spin rounded-full border-4 border-gray-300 border-t-gray-800"></div>
    </div>
  );
}

export function SpinnerWithText({ text }: { text?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-4">
      <div className="h-12 w-12 animate-spin rounded-full border-4 border-gray-300 border-t-gray-800"></div>
      {text && <p className="text-center text-sm text-gray-600">{text}</p>}
    </div>
  );
}
