import React, { useState, useRef, useEffect } from 'react'

interface SelectProps {
  children: React.ReactNode
  value?: string
  onValueChange?: (value: string) => void
  className?: string
}

interface SelectTriggerProps {
  children: React.ReactNode
  className?: string
}

interface SelectValueProps {
  placeholder?: string
  className?: string
}

interface SelectContentProps {
  children: React.ReactNode
  className?: string
}

interface SelectItemProps {
  children: React.ReactNode
  value: string
  className?: string
}

const SelectContext = React.createContext<{
  value: string
  onValueChange: (value: string) => void
  isOpen: boolean
  setIsOpen: (open: boolean) => void
}>({
  value: '',
  onValueChange: () => {},
  isOpen: false,
  setIsOpen: () => {}
})

export const Select: React.FC<SelectProps> = ({ children, value = '', onValueChange = () => {}, className = '' }) => {
  const [isOpen, setIsOpen] = useState(false)
  const selectRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (selectRef.current && !selectRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  return (
    <SelectContext.Provider value={{ value, onValueChange, isOpen, setIsOpen }}>
      <div ref={selectRef} className={`relative ${className}`}>
        {children}
      </div>
    </SelectContext.Provider>
  )
}

export const SelectTrigger: React.FC<SelectTriggerProps> = ({ children, className = '' }) => {
  const { isOpen, setIsOpen } = React.useContext(SelectContext)

  return (
    <button
      type="button"
      onClick={() => setIsOpen(!isOpen)}
      className={`flex h-10 w-full items-center justify-between rounded-md border border-gray-300 bg-white px-3 py-2 text-sm ring-offset-white placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
    >
      {children}
      <svg className="h-4 w-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
      </svg>
    </button>
  )
}

export const SelectValue: React.FC<SelectValueProps> = ({ placeholder = 'Select...', className = '' }) => {
  const { value } = React.useContext(SelectContext)

  return (
    <span className={`block truncate ${value ? '' : 'text-gray-500'} ${className}`}>
      {value || placeholder}
    </span>
  )
}

export const SelectContent: React.FC<SelectContentProps> = ({ children, className = '' }) => {
  const { isOpen } = React.useContext(SelectContext)

  if (!isOpen) return null

  return (
    <div className={`absolute z-50 mt-1 max-h-60 w-full overflow-auto rounded-md border bg-white py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none ${className}`}>
      {children}
    </div>
  )
}

export const SelectItem: React.FC<SelectItemProps> = ({ children, value, className = '' }) => {
  const { value: selectedValue, onValueChange, setIsOpen } = React.useContext(SelectContext)

  const handleClick = () => {
    onValueChange(value)
    setIsOpen(false)
  }

  const isSelected = selectedValue === value

  return (
    <button
      type="button"
      onClick={handleClick}
      className={`relative w-full cursor-pointer select-none py-2 pl-3 pr-9 text-left hover:bg-gray-100 ${isSelected ? 'bg-blue-50 text-blue-900' : 'text-gray-900'} ${className}`}
    >
      <span className={`block truncate ${isSelected ? 'font-medium' : 'font-normal'}`}>
        {children}
      </span>
      {isSelected && (
        <span className="absolute inset-y-0 right-0 flex items-center pr-4 text-blue-600">
          <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
          </svg>
        </span>
      )}
    </button>
  )
} 