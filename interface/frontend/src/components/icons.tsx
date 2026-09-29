import type { ReactNode, SVGProps } from 'react'

interface IconProps extends SVGProps<SVGSVGElement> {
  size?: number
}

function Icon({ size = 16, strokeWidth = 2, className = 'flex-none', children, ...rest }: IconProps & { children: ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={className}
      {...rest}
    >
      {children}
    </svg>
  )
}

export const MenuIcon = (p: IconProps) => <Icon size={26} strokeWidth={2.4} {...p}><path d="M4 6.5h16M4 12h16M4 17.5h16" /></Icon>
export const CloseIcon = (p: IconProps) => <Icon {...p}><path d="M6.5 6.5l11 11M17.5 6.5l-11 11" /></Icon>
export const PlusIcon = (p: IconProps) => <Icon {...p}><path d="M12 5.5v13M5.5 12h13" /></Icon>
export const CheckIcon = (p: IconProps) => <Icon {...p}><path d="M5 12.5l4.5 4.5L19 7.5" /></Icon>
export const ChevronDownIcon = (p: IconProps) => <Icon {...p}><path d="M6 9l6 6 6-6" /></Icon>

export const SearchIcon = (p: IconProps) => (
  <Icon size={24} strokeWidth={2.2} {...p}>
    <circle cx="10.5" cy="10.5" r="6.25" />
    <path d="M15.25 15.25L20 20" />
  </Icon>
)

export const SuggestionIcon = (p: IconProps) => (
  <Icon size={18} {...p}>
    <circle cx="10.5" cy="10.5" r="6" />
    <path d="M15 15l4.5 4.5" />
  </Icon>
)

export const FilterIcon = ({ filled = false, ...p }: IconProps & { filled?: boolean }) => (
  <Icon size={22} strokeWidth={1.8} {...p}>
    <path d="M3.5 5h17l-6.6 7.7v5.8l-3.8 2v-7.8z" fill={filled ? 'currentColor' : 'none'} />
  </Icon>
)

export function DocIcon({ format }: { format: string }) {
  return (
    <svg
      viewBox="0 0 36 44"
      aria-hidden="true"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      className="mt-px h-11 w-9 text-ink-2 max-sm:h-[37px] max-sm:w-[30px]"
    >
      <path d="M4.5 1.5H22.5L33.5 12.5V40.5a2 2 0 0 1-2 2H4.5a2 2 0 0 1-2-2V3.5a2 2 0 0 1 2-2Z" />
      <path d="M22.5 1.5V10.5a2 2 0 0 0 2 2h9" />
      <path d="M8 18h13M8 22.5h20M8 27h16" />
      <text
        x="18"
        y="37.5"
        textAnchor="middle"
        fill="currentColor"
        stroke="none"
        className="font-sans text-[8.5px] font-bold tracking-[.06em]"
      >
        {format}
      </text>
    </svg>
  )
}
