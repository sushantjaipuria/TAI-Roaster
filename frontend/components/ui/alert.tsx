import React from 'react'

interface AlertProps {
  children: React.ReactNode
  className?: string
  variant?: 'default' | 'success' | 'warning' | 'error' | 'destructive'
}

interface AlertTitleProps {
  children: React.ReactNode
  className?: string
}

interface AlertDescriptionProps {
  children: React.ReactNode
  className?: string
}

export const Alert: React.FC<AlertProps> = ({ children, className = '', variant = 'default' }) => {
  const variantClasses = {
    default: 'border-gray-200 bg-gray-50 text-gray-700',
    success: 'border-green-200 bg-green-50 text-green-700',
    warning: 'border-yellow-200 bg-yellow-50 text-yellow-700',
    error: 'border-red-200 bg-red-50 text-red-700',
    destructive: 'border-red-200 bg-red-50 text-red-700'
  }

  return (
    <div className={`rounded-lg border p-4 ${variantClasses[variant]} ${className}`}>
      {children}
    </div>
  )
}

export const AlertTitle: React.FC<AlertTitleProps> = ({ children, className = '' }) => {
  return (
    <h5 className={`mb-1 font-medium leading-none tracking-tight ${className}`}>
      {children}
    </h5>
  )
}

export const AlertDescription: React.FC<AlertDescriptionProps> = ({ children, className = '' }) => {
  return (
    <div className={`text-sm [&_p]:leading-relaxed ${className}`}>
      {children}
    </div>
  )
} 