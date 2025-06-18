import React from 'react'
import { 
  LineChart as RechartsLineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  ReferenceLine
} from 'recharts'
import { format, parseISO } from 'date-fns'

interface LineChartData {
  date: string
  [key: string]: string | number
}

interface LineConfig {
  dataKey: string
  name: string
  color: string
  strokeWidth?: number
  strokeDasharray?: string
}

interface LineChartProps {
  data: LineChartData[]
  lines: LineConfig[]
  title?: string
  height?: number
  xAxisLabel?: string
  yAxisLabel?: string
  showGrid?: boolean
  showLegend?: boolean
  formatXAxis?: (value: string) => string
  formatYAxis?: (value: number) => string
  formatTooltip?: (value: number, name: string) => string
}

// Default tooltip formatter
const defaultTooltipFormatter = (value: number, name: string) => {
  if (name.toLowerCase().includes('return') || name.toLowerCase().includes('performance')) {
    return [`${value.toFixed(2)}%`, name]
  }
  if (name.toLowerCase().includes('value') || name.toLowerCase().includes('amount')) {
    return [`₹${value.toLocaleString('en-IN')}`, name]
  }
  return [value.toFixed(2), name]
}

// Custom tooltip component
const CustomTooltip = ({ active, payload, label, formatTooltip }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-lg">
        <p className="font-medium text-gray-900 mb-2">
          {format(parseISO(label), 'MMM d, yyyy')}
        </p>
        {payload.map((entry: any, index: number) => (
          <p key={index} className="text-sm" style={{ color: entry.color }}>
            <span className="font-medium">
              {formatTooltip ? formatTooltip(entry.value, entry.name) : defaultTooltipFormatter(entry.value, entry.name)[1]}:
            </span>
            <span className="ml-2">
              {formatTooltip ? formatTooltip(entry.value, entry.name) : defaultTooltipFormatter(entry.value, entry.name)[0]}
            </span>
          </p>
        ))}
      </div>
    )
  }
  return null
}

export default function LineChart({
  data,
  lines,
  title,
  height = 400,
  xAxisLabel,
  yAxisLabel,
  showGrid = true,
  showLegend = true,
  formatXAxis,
  formatYAxis,
  formatTooltip
}: LineChartProps) {
  
  // Default X-axis formatter
  const defaultXAxisFormatter = (value: string) => {
    try {
      return format(parseISO(value), 'MMM yyyy')
    } catch {
      return value
    }
  }

  // Default Y-axis formatter
  const defaultYAxisFormatter = (value: number) => {
    if (Math.abs(value) >= 1000000) {
      return `₹${(value / 1000000).toFixed(1)}M`
    }
    if (Math.abs(value) >= 1000) {
      return `₹${(value / 1000).toFixed(0)}K`
    }
    return value.toFixed(0)
  }

  return (
    <div className="w-full">
      {title && (
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
      )}
      
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <RechartsLineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            {showGrid && (
              <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
            )}
            
            <XAxis
              dataKey="date"
              tickFormatter={formatXAxis || defaultXAxisFormatter}
              tick={{ fontSize: 12, fill: '#6b7280' }}
              axisLine={{ stroke: '#d1d5db' }}
              tickLine={{ stroke: '#d1d5db' }}
            />
            
            <YAxis
              tickFormatter={formatYAxis || defaultYAxisFormatter}
              tick={{ fontSize: 12, fill: '#6b7280' }}
              axisLine={{ stroke: '#d1d5db' }}
              tickLine={{ stroke: '#d1d5db' }}
            />
            
            <Tooltip 
              content={<CustomTooltip formatTooltip={formatTooltip} />}
              cursor={{ strokeDasharray: '3 3', stroke: '#d1d5db' }}
            />
            
            {showLegend && (
              <Legend 
                wrapperStyle={{ paddingTop: '20px' }}
                iconType="line"
              />
            )}

            {/* Zero reference line for returns */}
            {lines.some(line => line.name.toLowerCase().includes('return')) && (
              <ReferenceLine y={0} stroke="#9ca3af" strokeDasharray="2 2" />
            )}
            
            {lines.map((lineConfig, index) => (
              <Line
                key={lineConfig.dataKey}
                type="monotone"
                dataKey={lineConfig.dataKey}
                name={lineConfig.name}
                stroke={lineConfig.color}
                strokeWidth={lineConfig.strokeWidth || 2}
                strokeDasharray={lineConfig.strokeDasharray}
                dot={{ r: 0 }}
                activeDot={{ r: 4, fill: lineConfig.color }}
                connectNulls={false}
              />
            ))}
          </RechartsLineChart>
        </ResponsiveContainer>
      </div>

      {/* Labels */}
      {(xAxisLabel || yAxisLabel) && (
        <div className="flex justify-between mt-2 text-xs text-gray-500">
          <span>{xAxisLabel}</span>
          <span>{yAxisLabel}</span>
        </div>
      )}
    </div>
  )
} 