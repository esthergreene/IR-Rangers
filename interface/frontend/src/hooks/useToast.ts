import { useRef, useState } from 'react'

export function useToast(duration = 2800) {
  const [message, setMessage] = useState('')
  const [visible, setVisible] = useState(false)
  const timer = useRef<number>(undefined)

  function show(next: string) {
    setMessage(next)
    setVisible(true)
    clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setVisible(false), duration)
  }

  return { message, visible, show }
}
