import { useState, useRef, useEffect } from 'react'
import { ChevronDown } from 'lucide-react'

interface CustomSelectProps {
  value: string
  onChange: (value: string) => void
  options: string[]
  placeholder: string
  disabled?: boolean
  optionDetails?: Record<string, string>
}

export default function CustomSelect({
  value,
  onChange,
  options,
  placeholder,
  disabled = false,
  optionDetails = {},
}: CustomSelectProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [hoveredOption, setHoveredOption] = useState<string | null>(null)
  const [tooltipPosition, setTooltipPosition] = useState<{ top: number; left: number } | null>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)
  const optionRefs = useRef<Map<string, HTMLButtonElement>>(new Map())

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [isOpen])

  useEffect(() => {
    if (!isOpen) {
      setHoveredOption(null)
      setTooltipPosition(null)
    }
  }, [isOpen])

  useEffect(() => {
    if (hoveredOption) {
      const updatePosition = () => {
        const buttonElement = optionRefs.current.get(hoveredOption)
        if (buttonElement) {
          const rect = buttonElement.getBoundingClientRect()
          // Position tooltip vertically centered on the button, to the right
          setTooltipPosition({
            top: rect.top + rect.height / 2,
            left: rect.right + 8
          })
        }
      }
      
      updatePosition()
      // Update position on scroll/resize
      window.addEventListener('scroll', updatePosition, true)
      window.addEventListener('resize', updatePosition)
      
      return () => {
        window.removeEventListener('scroll', updatePosition, true)
        window.removeEventListener('resize', updatePosition)
      }
    } else {
      setTooltipPosition(null)
    }
  }, [hoveredOption])

  const selectedValue = value || placeholder

  return (
    <>
      <div className="relative z-40" ref={dropdownRef}>
        <div className="relative">
          <button
            type="button"
            onClick={() => !disabled && setIsOpen(!isOpen)}
            disabled={disabled}
            className={`w-full p-2 border-2 border-black bg-background text-left flex items-center justify-between disabled:opacity-50 shadow-brutal transition-colors ${
              isOpen || value 
                ? 'border-primary bg-primary text-primary-foreground hover:border-accent hover:bg-accent hover:text-accent-foreground' 
                : ''
            } ${disabled ? '' : !isOpen && !value ? 'hover:border-accent hover:bg-accent hover:text-accent-foreground' : ''}`}
          >
            <span className={`${isOpen || value ? 'text-primary-foreground' : value ? 'text-foreground' : 'text-muted-foreground'} ${disabled ? 'text-muted-foreground' : ''}`}>
              {selectedValue}
            </span>
            <ChevronDown 
              className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
            />
          </button>
        </div>
        
        {isOpen && !disabled && (
          <div className="w-full mt-1 border-2 border-black bg-background shadow-brutal max-h-60 overflow-y-auto relative z-50">
            {options.map((option) => (
              <div
                key={option}
                className="relative group"
                onMouseEnter={() => setHoveredOption(option)}
                onMouseLeave={() => setHoveredOption(null)}
              >
                <button
                  ref={(el) => {
                    if (el) {
                      optionRefs.current.set(option, el)
                    } else {
                      optionRefs.current.delete(option)
                    }
                  }}
                  type="button"
                  onClick={() => {
                    onChange(option)
                    setIsOpen(false)
                  }}
                  onFocus={() => setHoveredOption(option)}
                  onBlur={() => setHoveredOption(null)}
                  className={`w-full p-2 text-left border-b border-black last:border-b-0 transition-colors ${
                    value === option
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-background hover:bg-accent hover:text-accent-foreground'
                  }`}
                >
                  {option}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
      {tooltipPosition && hoveredOption && optionDetails[hoveredOption] && (
        <div 
          className="fixed w-64 border-2 border-black bg-black text-white text-xs leading-relaxed shadow-brutal px-3 py-2 pointer-events-none z-[210] whitespace-normal"
          style={{
            top: `${tooltipPosition.top}px`,
            left: `${tooltipPosition.left}px`,
            transform: 'translateY(-50%)'
          }}
        >
          <p className="font-semibold text-sm text-white mb-1">{hoveredOption}</p>
          <p>{optionDetails[hoveredOption]}</p>
        </div>
      )}
    </>
  )
}
