import type { ButtonHTMLAttributes } from 'react'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'primary'
  size?: 'md' | 'sm'
}

const VARIANTS = {
  default: 'border-rule-strong bg-paper text-ink hover:border-ink-2',
  primary: 'border-primary bg-primary text-on-primary hover:border-primary-hover hover:bg-primary-hover',
}

const SIZES = {
  md: 'min-h-9 px-4',
  sm: 'min-h-8 px-3',
}

export function Button({ variant = 'default', size = 'md', className = '', ...rest }: ButtonProps) {
  return (
    <button
      type="button"
      className={`inline-flex items-center justify-center rounded-lg border text-[14px] font-semibold whitespace-nowrap ${VARIANTS[variant]} ${SIZES[size]} ${className}`}
      {...rest}
    />
  )
}

export function TextButton({ className = '', ...rest }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      className={`px-0.5 py-1.5 text-[14px] font-medium text-ink-2 underline decoration-rule-strong underline-offset-3 hover:text-ink hover:decoration-current ${className}`}
      {...rest}
    />
  )
}
