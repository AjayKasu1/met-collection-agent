export function CollectionMark(): React.ReactNode {
  return (
    <svg aria-hidden="true" className="collection-mark" viewBox="0 0 40 40">
      <path d="M7 7h26v26H7z" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <path d="M13 7v26M27 7v26M7 13h26M7 27h26" stroke="currentColor" strokeWidth="1" />
      <circle cx="20" cy="20" r="4" fill="currentColor" />
    </svg>
  );
}
