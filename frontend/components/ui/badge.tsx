import React from 'react'

interface BadgeProps {
  children: React.ReactNode
  className?: string
  variant?: 'default' | 'success' | 'warning' | 'error' | 'secondary'
}

export const Badge: React.FC<BadgeProps> = ({ children, className = '', variant = 'default' }) => {
  const variantClasses = {
    default: 'bg-blue-100 text-blue-800',
    success: 'bg-green-100 text-green-800',
    warning: 'bg-yellow-100 text-yellow-800',
    error: 'bg-red-100 text-red-800',
    secondary: 'bg-gray-100 text-gray-800'
  }

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${variantClasses[variant]} ${className}`}>
      {children}
    </span>
  )
} 