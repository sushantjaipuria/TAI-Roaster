import { type ClassValue, clsx } from 'clsx'

// Utility function to merge classes (useful with Tailwind)
export function cn(...inputs: ClassValue[]) {
  return clsx(inputs)
}

// Format currency values
export function formatCurrency(amount: number, currency: string = 'USD'): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(amount)
}

// Format percentage values
export function formatPercentage(value: number, decimals: number = 2): string {
  return `${(value * 100).toFixed(decimals)}%`
}

// Format large numbers with K, M, B suffixes
export function formatCompactNumber(num: number): string {
  if (num >= 1e9) {
    return `${(num / 1e9).toFixed(1)}B`
  } else if (num >= 1e6) {
    return `${(num / 1e6).toFixed(1)}M`
  } else if (num >= 1e3) {
    return `${(num / 1e3).toFixed(1)}K`
  }
  return num.toString()
}

// Validate email
export function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return emailRegex.test(email)
}

// Validate ticker symbol
export function isValidTicker(ticker: string): boolean {
  const tickerRegex = /^[A-Z]{1,5}$/
  return tickerRegex.test(ticker.toUpperCase())
}

// Generate random ID
export function generateId(): string {
  return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15)
}

// Debounce function
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout)
    timeout = setTimeout(() => func(...args), wait)
  }
}

// Calculate portfolio allocation percentages
export function calculateAllocation(items: Array<{ totalValue: number }>, totalValue: number) {
  return items.map(item => ({
    ...item,
    allocation: totalValue > 0 ? (item.totalValue / totalValue) * 100 : 0
  }))
}

// Get risk level color
export function getRiskColor(riskLevel: 'low' | 'medium' | 'high'): string {
  switch (riskLevel) {
    case 'low':
      return 'text-success-600 bg-success-50'
    case 'medium':
      return 'text-warning-600 bg-warning-50'
    case 'high':
      return 'text-danger-600 bg-danger-50'
    default:
      return 'text-gray-600 bg-gray-50'
  }
}

// Get recommendation color
export function getRecommendationColor(recommendation: 'buy' | 'hold' | 'sell'): string {
  switch (recommendation) {
    case 'buy':
      return 'text-success-600 bg-success-50'
    case 'hold':
      return 'text-warning-600 bg-warning-50'
    case 'sell':
      return 'text-danger-600 bg-danger-50'
    default:
      return 'text-gray-600 bg-gray-50'
  }
}

// Sleep function for delays
export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

// Local storage helpers
export const storage = {
  get: (key: string) => {
    if (typeof window === 'undefined') return null
    try {
      const item = window.localStorage.getItem(key)
      return item ? JSON.parse(item) : null
    } catch {
      return null
    }
  },
  set: (key: string, value: any) => {
    if (typeof window === 'undefined') return
    try {
      window.localStorage.setItem(key, JSON.stringify(value))
    } catch {
      // Handle storage errors silently
    }
  },
  remove: (key: string) => {
    if (typeof window === 'undefined') return
    try {
      window.localStorage.removeItem(key)
    } catch {
      // Handle storage errors silently
    }
  }
} 