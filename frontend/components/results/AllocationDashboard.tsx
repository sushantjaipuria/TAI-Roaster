import React, { useState } from 'react'
import { AlertTriangle, TrendingUp, Shield, BarChart3 } from 'lucide-react'
import { ResultsComponentProps } from '../../lib/types-results'
import PieChart from '../charts/PieChart'

export default function AllocationDashboard({ data }: ResultsComponentProps) {
  const [activeTab, setActiveTab] = useState<'sectors' | 'marketCap' | 'assetTypes'>('sectors')

  // Convert allocation data to chart format
  const sectorData = Object.entries(data.allocation.current.sectors).map(([name, value]) => ({
    name,
    value: Number(value)
  }))

  const marketCapData = Object.entries(data.allocation.current.marketCap).map(([name, value]) => ({
    name: name.charAt(0).toUpperCase() + name.slice(1),
    value: Number(value)
  }))

  const assetTypeData = Object.entries(data.allocation.current.assetTypes).map(([name, value]) => ({
    name,
    value: Number(value)
  }))

  // Get concentration flags
  const concentrationFlags = data.allocation.concentration.riskFlags

  // Calculate correlation risk level
  const avgCorrelation = data.allocation.correlation.averageCorrelation
  const correlationRisk = avgCorrelation > 0.7 ? 'HIGH' : avgCorrelation > 0.5 ? 'MEDIUM' : 'LOW'

  // Chart tab configurations
  const chartTabs = [
    { 
      id: 'sectors' as const, 
      label: 'Sector Allocation', 
      data: sectorData,
      description: 'Distribution across industry sectors'
    },
    { 
      id: 'marketCap' as const, 
      label: 'Market Cap', 
      data: marketCapData,
      description: 'Large, mid, and small cap distribution'
    },
    { 
      id: 'assetTypes' as const, 
      label: 'Asset Classes', 
      data: assetTypeData,
      description: 'Equity, debt, and cash allocation'
    }
  ]

  const activeChartData = chartTabs.find(tab => tab.id === activeTab)

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-gray-900">📊 Asset Allocation & Diversification</h2>
          <p className="text-sm text-gray-600 mt-1">
            Diagnose diversification, exposure, and correlation risks
          </p>
        </div>
        
        {/* Diversification Score */}
        <div className="flex items-center gap-2 px-4 py-2 bg-blue-100 rounded-lg">
          <BarChart3 className="w-4 h-4 text-blue-600" />
          <span className="text-sm font-medium text-blue-800">Diversification Score</span>
          <span className="text-xl font-bold text-blue-600">{Math.round(data.diversificationScore)}</span>
          <span className="text-xs text-blue-600">/100</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Charts Section */}
        <div className="lg:col-span-2">
          {/* Chart Tabs */}
          <div className="flex flex-wrap gap-2 mb-6">
            {chartTabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Active Chart */}
          {activeChartData && (
            <div className="bg-gray-50 rounded-lg p-6">
              <div className="mb-4">
                <h3 className="text-lg font-semibold text-gray-900">{activeChartData.label}</h3>
                <p className="text-sm text-gray-600">{activeChartData.description}</p>
              </div>
              
              <PieChart
                data={activeChartData.data}
                height={350}
                showLegend={true}
              />
            </div>
          )}
        </div>

        {/* Risk Analysis Section */}
        <div className="lg:col-span-1 space-y-6">
          {/* Concentration Flags */}
          <div className="bg-yellow-50 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <AlertTriangle className="w-5 h-5 text-yellow-600" />
              <h3 className="text-lg font-semibold text-yellow-800">Concentration Risks</h3>
            </div>
            
            <div className="space-y-3">
              {concentrationFlags.length > 0 ? (
                concentrationFlags.map((flag, index) => (
                  <div key={index} className="flex items-start gap-2">
                    <div className={`w-2 h-2 rounded-full mt-2 flex-shrink-0 ${
                      flag.severity === 'HIGH' ? 'bg-red-500' : 
                      flag.severity === 'MEDIUM' ? 'bg-yellow-500' : 'bg-green-500'
                    }`}></div>
                    <div>
                      <div className="text-sm font-medium text-gray-900">{flag.type}</div>
                      <div className="text-xs text-gray-600">{flag.message}</div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-sm text-green-700">
                  No significant concentration risks detected
                </div>
              )}
            </div>
          </div>

          {/* Correlation Analysis */}
          <div className="bg-blue-50 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <Shield className="w-5 h-5 text-blue-600" />
              <h3 className="text-lg font-semibold text-blue-800">Correlation Analysis</h3>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-700">Average Correlation</span>
                <span className="text-sm font-medium">{avgCorrelation.toFixed(2)}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-700">Risk Level</span>
                <span className={`text-sm font-medium px-2 py-1 rounded ${
                  correlationRisk === 'HIGH' ? 'bg-red-100 text-red-800' :
                  correlationRisk === 'MEDIUM' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-green-100 text-green-800'
                }`}>
                  {correlationRisk}
                </span>
              </div>

              {/* Highly Correlated Pairs */}
              {data.allocation.correlation.highlyCorrelatedPairs.length > 0 && (
                <div className="mt-3 pt-3 border-t border-blue-200">
                  <div className="text-xs text-blue-700 mb-2">Highly Correlated Pairs:</div>
                  {data.allocation.correlation.highlyCorrelatedPairs.slice(0, 3).map((pair, index) => (
                    <div key={index} className="text-xs text-gray-600 flex justify-between">
                      <span>{pair.stock1} ↔ {pair.stock2}</span>
                      <span className="font-medium">{pair.correlation.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Sector Concentration Score */}
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <TrendingUp className="w-5 h-5 text-gray-600" />
              <h3 className="text-lg font-semibold text-gray-800">Sector Analysis</h3>
            </div>
            
            <div className="space-y-3">
              {data.allocation.concentration.sectorConcentration.slice(0, 3).map((sector, index) => (
                <div key={index} className="space-y-1">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-gray-900">{sector.sector}</span>
                    <span className={`text-sm font-medium ${
                      sector.isOverweight ? 'text-orange-600' : 'text-green-600'
                    }`}>
                      {sector.allocation.toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${
                          sector.isOverweight ? 'bg-orange-500' : 'bg-green-500'
                        }`}
                        style={{ width: `${Math.min(100, (sector.allocation / 30) * 100)}%` }}
                      ></div>
                    </div>
                    <span className="text-xs text-gray-500">
                      vs {sector.benchmark.toFixed(1)}%
                    </span>
                  </div>
                  
                  {sector.isOverweight && (
                    <div className="text-xs text-orange-600">
                      Overweight vs benchmark
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Quick Recommendations */}
          <div className="bg-green-50 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-green-800 mb-3">Quick Actions</h3>
            <div className="space-y-2 text-sm text-green-700">
              {data.diversificationScore < 70 && (
                <div>• Consider diversifying across more sectors</div>
              )}
              {data.allocation.current.marketCap.largeCap > 70 && (
                <div>• Add mid-cap exposure for growth potential</div>
              )}
              {avgCorrelation > 0.6 && (
                <div>• Reduce correlation by adding defensive stocks</div>
              )}
              {data.allocation.current.assetTypes.Cash > 10 && (
                <div>• Deploy excess cash for better returns</div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Summary */}
      <div className="mt-8 pt-6 border-t border-gray-200">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{data.allocation.concentration.topHoldingsPct.toFixed(0)}%</div>
            <div className="text-sm text-gray-600">Top 5 Holdings</div>
            <div className="text-xs text-gray-500 mt-1">
              {data.allocation.concentration.topHoldingsPct > 50 ? 'High concentration' : 'Well distributed'}
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{sectorData.length}</div>
            <div className="text-sm text-gray-600">Sectors Covered</div>
            <div className="text-xs text-gray-500 mt-1">
              {sectorData.length >= 6 ? 'Good diversification' : 'Consider more sectors'}
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">{avgCorrelation.toFixed(2)}</div>
            <div className="text-sm text-gray-600">Avg Correlation</div>
            <div className="text-xs text-gray-500 mt-1">
              {avgCorrelation < 0.5 ? 'Low correlation' : avgCorrelation < 0.7 ? 'Moderate correlation' : 'High correlation'}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 