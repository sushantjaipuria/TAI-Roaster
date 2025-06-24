import React from 'react'
import { TrendingUp, Shield, BarChart3, AlertTriangle, CheckCircle, AlertCircle } from 'lucide-react'
import { ResultsComponentProps } from '../../lib/types-results'

// Circular Progress Component (WHOOP-style)
const CircularProgress = ({ value, max = 100, size = 120, strokeWidth = 8, color = 'blue' }: {
  value: number
  max?: number
  size?: number
  strokeWidth?: number
  color?: 'blue' | 'green' | 'yellow' | 'red'
}) => {
  const radius = (size - strokeWidth) / 2
  const circumference = 2 * Math.PI * radius
  const percentage = (value / max) * 100
  const strokeDasharray = circumference
  const strokeDashoffset = circumference - (percentage / 100) * circumference

  const colorClasses = {
    blue: 'stroke-blue-500',
    green: 'stroke-green-500',
    yellow: 'stroke-yellow-500',
    red: 'stroke-red-500'
  }

  const bgColorClasses = {
    blue: 'stroke-blue-100',
    green: 'stroke-green-100',
    yellow: 'stroke-yellow-100',
    red: 'stroke-red-100'
  }

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          className={bgColorClasses[color]}
          strokeWidth={strokeWidth}
        />
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          className={colorClasses[color]}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={strokeDasharray}
          strokeDashoffset={strokeDashoffset}
          style={{
            transition: 'stroke-dashoffset 0.5s ease-in-out'
          }}
        />
      </svg>
      {/* Center text */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-900">{Math.round(value)}</div>
          <div className="text-xs text-gray-500">/{max}</div>
        </div>
      </div>
    </div>
  )
}

// Health Score Card Component
const HealthScoreCard = ({ 
  title, 
  score, 
  subtitle, 
  icon, 
  color,
  status,
  insight
}: {
  title: string
  score: number
  subtitle: string
  icon: React.ReactNode
  color: 'blue' | 'green' | 'yellow' | 'red'
  status: 'excellent' | 'good' | 'fair' | 'poor'
  insight: string
}) => {
  const statusColors = {
    excellent: 'bg-green-50 border-green-200',
    good: 'bg-blue-50 border-blue-200',
    fair: 'bg-yellow-50 border-yellow-200',
    poor: 'bg-red-50 border-red-200'
  }

  const statusIcons = {
    excellent: <CheckCircle className="w-5 h-5 text-green-600" />,
    good: <CheckCircle className="w-5 h-5 text-blue-600" />,
    fair: <AlertCircle className="w-5 h-5 text-yellow-600" />,
    poor: <AlertTriangle className="w-5 h-5 text-red-600" />
  }

  return (
    <div className={`relative p-6 rounded-xl border-2 ${statusColors[status]} transition-all duration-200 hover:scale-105`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          {icon}
          <div>
            <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
            <p className="text-sm text-gray-600">{subtitle}</p>
          </div>
        </div>
        {statusIcons[status]}
      </div>

      {/* Score Circle */}
      <div className="flex justify-center mb-4">
        <CircularProgress value={score} color={color} />
      </div>

      {/* Insight */}
      <div className="text-center">
        <p className="text-sm text-gray-700 font-medium">{insight}</p>
      </div>
    </div>
  )
}

// Key Action Card
const KeyActionCard = ({ data }: { data: any }) => {
  const getTopAction = () => {
    if (data.actionPlan.improvements.length > 0) {
      const highPriority = data.actionPlan.improvements.find((item: any) => item.priority === 'HIGH')
      return highPriority || data.actionPlan.improvements[0]
    }
    return null
  }

  const action = getTopAction()

  if (!action) {
    return (
      <div className="bg-green-50 border-2 border-green-200 rounded-xl p-6">
        <div className="flex items-center gap-3 mb-3">
          <CheckCircle className="w-6 h-6 text-green-600" />
          <h3 className="text-lg font-semibold text-green-900">Portfolio Optimized</h3>
        </div>
        <p className="text-green-700">Your portfolio is well-balanced and performing optimally. Continue monitoring market conditions.</p>
      </div>
    )
  }

  return (
    <div className="bg-blue-50 border-2 border-blue-200 rounded-xl p-6">
      <div className="flex items-center gap-3 mb-3">
        <TrendingUp className="w-6 h-6 text-blue-600" />
        <h3 className="text-lg font-semibold text-blue-900">Key Action</h3>
      </div>
      <p className="text-blue-700 font-medium mb-2">{action.suggestion}</p>
      <p className="text-sm text-blue-600">{action.expectedBenefit}</p>
    </div>
  )
}

export default function PortfolioHealthDashboard({ data }: ResultsComponentProps) {
  // Calculate health scores with insights
  const getHealthScore = () => {
    const score = data.rating.taiScore
    if (score >= 80) return { status: 'excellent' as const, insight: 'Outstanding portfolio performance' }
    if (score >= 65) return { status: 'good' as const, insight: 'Strong portfolio fundamentals' }
    if (score >= 50) return { status: 'fair' as const, insight: 'Room for optimization' }
    return { status: 'poor' as const, insight: 'Requires immediate attention' }
  }

  const getRiskScore = () => {
    const riskScore = data.rating.riskManagement
    if (riskScore >= 75) return { status: 'excellent' as const, insight: 'Well-managed risk profile' }
    if (riskScore >= 60) return { status: 'good' as const, insight: 'Balanced risk approach' }
    if (riskScore >= 40) return { status: 'fair' as const, insight: 'Consider risk reduction' }
    return { status: 'poor' as const, insight: 'High risk exposure detected' }
  }

  const getDiversificationScore = () => {
    const score = data.diversificationScore
    if (score >= 85) return { status: 'excellent' as const, insight: 'Excellently diversified' }
    if (score >= 70) return { status: 'good' as const, insight: 'Well-diversified portfolio' }
    if (score >= 50) return { status: 'fair' as const, insight: 'Needs more diversification' }
    return { status: 'poor' as const, insight: 'Concentrated portfolio risk' }
  }

  const healthStatus = getHealthScore()
  const riskStatus = getRiskScore()
  const diversificationStatus = getDiversificationScore()

  return (
    <div className="bg-white">
      {/* Hero Header */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 text-white py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-3xl font-bold mb-2">Portfolio Health Dashboard</h1>
            <p className="text-blue-100 text-lg">Your financial wellness at a glance</p>
          </div>
        </div>
      </div>

      {/* Core Health Scores */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
          {/* Portfolio Health Score */}
          <HealthScoreCard
            title="Portfolio Health"
            score={data.rating.taiScore}
            subtitle="Overall performance rating"
            icon={<TrendingUp className="w-6 h-6 text-blue-600" />}
            color="blue"
            status={healthStatus.status}
            insight={healthStatus.insight}
          />

          {/* Risk Score */}
          <HealthScoreCard
            title="Risk Management"
            score={data.rating.riskManagement}
            subtitle="Risk-adjusted assessment"
            icon={<Shield className="w-6 h-6 text-green-600" />}
            color={riskStatus.status === 'excellent' ? 'green' : riskStatus.status === 'poor' ? 'red' : 'yellow'}
            status={riskStatus.status}
            insight={riskStatus.insight}
          />

          {/* Diversification Score */}
          <HealthScoreCard
            title="Diversification"
            score={data.diversificationScore}
            subtitle="Portfolio balance score"
            icon={<BarChart3 className="w-6 h-6 text-purple-600" />}
            color={diversificationStatus.status === 'excellent' ? 'green' : diversificationStatus.status === 'poor' ? 'red' : 'blue'}
            status={diversificationStatus.status}
            insight={diversificationStatus.insight}
          />
        </div>

        {/* Key Action Section */}
        <KeyActionCard data={data} />
      </div>
    </div>
  )
} 