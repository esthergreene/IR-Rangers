export function Toast({ message, visible }: { message: string; visible: boolean }) {
  return (
    <div
      role="status"
      aria-live="polite"
      className={`fixed bottom-6 left-1/2 z-60 max-w-[calc(100vw-32px)] -translate-x-1/2 rounded-lg bg-primary px-4 py-2.5 text-[14px] text-on-primary transition-[opacity,translate] duration-200 ${
        visible ? 'translate-y-0 opacity-100' : 'pointer-events-none translate-y-3 opacity-0'
      }`}
    >
      {message}
    </div>
  )
}
