import React from 'react'
import { CheckCircle, AlertCircle, TrendingUp, Target, Clock, Zap, Shield } from 'lucide-react'
import { ResultsComponentProps } from '../../lib/types-results'

export default function ActionPlan({ data }: ResultsComponentProps) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-gray-900">🤮 Pros, Cons, and Action Plan</h2>
          <p className="text-sm text-gray-600 mt-1">
            Clear, actionable steps with ROI impact and priority ranking
          </p>
        </div>
        
        {/* Summary Stats */}
        <div className="text-right">
          <div className="text-2xl font-bold text-blue-600">{data.actionPlan.summary.totalActions}</div>
          <div className="text-sm text-gray-600">Total Actions</div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Pros */}
        <div className="bg-green-50 rounded-lg p-6">
          <div className="flex items-center gap-2 mb-4">
            <CheckCircle className="w-5 h-5 text-green-600" />
            <h3 className="text-lg font-semibold text-green-800">Portfolio Strengths</h3>
          </div>
          
          <div className="space-y-3">
            {data.actionPlan.pros.map((pro, index) => (
              <div key={index} className="border-l-2 border-green-300 pl-3">
                <div className="text-sm font-medium text-green-800">{pro.category}</div>
                <div className="text-sm text-green-700">{pro.achievement}</div>
                <div className="text-xs text-green-600 mt-1">{pro.impact}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Cons */}
        <div className="bg-red-50 rounded-lg p-6">
          <div className="flex items-center gap-2 mb-4">
            <AlertCircle className="w-5 h-5 text-red-600" />
            <h3 className="text-lg font-semibold text-red-800">Areas of Concern</h3>
          </div>
          
          <div className="space-y-3">
            {data.actionPlan.cons.map((con, index) => (
              <div key={index} className="border-l-2 border-red-300 pl-3">
                <div className="flex items-center gap-2">
                  <div className="text-sm font-medium text-red-800">{con.category}</div>
                  <span className={`text-xs px-2 py-1 rounded ${
                    con.severity === 'HIGH' ? 'bg-red-200 text-red-800' :
                    con.severity === 'MEDIUM' ? 'bg-yellow-200 text-yellow-800' :
                    'bg-gray-200 text-gray-800'
                  }`}>
                    {con.severity}
                  </span>
                </div>
                <div className="text-sm text-red-700">{con.issue}</div>
                <div className="text-xs text-red-600 mt-1">{con.impact}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Areas of Improvement */}
        <div className="bg-blue-50 rounded-lg p-6">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-blue-600" />
            <h3 className="text-lg font-semibold text-blue-800">Improvement Areas</h3>
          </div>
          
          <div className="space-y-3">
            {data.actionPlan.improvements.map((improvement, index) => (
              <div key={index} className="border-l-2 border-blue-300 pl-3">
                <div className="flex items-center gap-2">
                  <div className="text-sm font-medium text-blue-800">{improvement.area}</div>
                  <span className={`text-xs px-2 py-1 rounded ${
                    improvement.priority === 'HIGH' ? 'bg-blue-200 text-blue-800' :
                    improvement.priority === 'MEDIUM' ? 'bg-blue-100 text-blue-700' :
                    'bg-gray-100 text-gray-700'
                  }`}>
                    {improvement.priority}
                  </span>
                </div>
                <div className="text-sm text-blue-700">{improvement.suggestion}</div>
                <div className="text-xs text-blue-600 mt-1">{improvement.expectedBenefit}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Action Items */}
      <div className="mt-8 pt-6 border-t border-gray-200">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900">Top Priority Actions</h3>
          <div className="flex items-center gap-4 text-sm text-gray-600">
            <div className="flex items-center gap-1">
              <Target className="w-4 h-4" />
              <span>Expected Return: +{data.actionPlan.summary.expectedReturnImprovement.toFixed(1)}%</span>
            </div>
            <div className="flex items-center gap-1">
              <Shield className="w-4 h-4" />
              <span>Risk Reduction: -{data.actionPlan.summary.riskReduction.toFixed(0)}%</span>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          {data.actionPlan.actions.map((action, index) => (
            <div key={action.id} className={`border-2 rounded-lg p-4 ${
              action.priority === 'HIGH' ? 'border-red-200 bg-red-50' :
              action.priority === 'MEDIUM' ? 'border-yellow-200 bg-yellow-50' :
              'border-blue-200 bg-blue-50'
            }`}>
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                      action.priority === 'HIGH' ? 'bg-red-600 text-white' :
                      action.priority === 'MEDIUM' ? 'bg-yellow-600 text-white' :
                      'bg-blue-600 text-white'
                    }`}>
                      {action.priority}
                    </span>
                    
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      action.type === 'BUY' ? 'bg-green-100 text-green-800' :
                      action.type === 'SELL' ? 'bg-red-100 text-red-800' :
                      action.type === 'TRIM' ? 'bg-orange-100 text-orange-800' :
                      'bg-gray-100 text-gray-800'
                    }`}>
                      {action.type}
                    </span>
                    
                    <span className="text-xs text-gray-500 uppercase tracking-wide">
                      {action.category.replace('_', ' ')}
                    </span>
                  </div>
                  
                  <h4 className="text-base font-semibold text-gray-900 mb-1">
                    {action.description}
                  </h4>
                  
                  <p className="text-sm text-gray-700 mb-3">
                    {action.reasoning}
                  </p>

                  {/* Action Details */}
                  {(action.currentAllocation || action.targetAllocation) && (
                    <div className="text-sm text-gray-600 mb-2">
                      {action.currentAllocation && action.targetAllocation && (
                        <span>
                          Allocation: {action.currentAllocation.toFixed(1)}% → {action.targetAllocation.toFixed(1)}%
                        </span>
                      )}
                    </div>
                  )}
                </div>
                
                <div className="ml-4 text-right">
                  {/* Impact Metrics */}
                  <div className="space-y-1">
                    {action.expectedReturn !== 0 && (
                      <div className={`text-sm font-medium ${
                        action.expectedReturn > 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {action.expectedReturn > 0 ? '+' : ''}{action.expectedReturn.toFixed(1)}% return
                      </div>
                    )}
                    
                    {action.riskImpact !== 0 && (
                      <div className={`text-sm font-medium ${
                        action.riskImpact < 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {action.riskImpact > 0 ? '+' : ''}{action.riskImpact.toFixed(0)}% risk
                      </div>
                    )}
                  </div>
                  
                  {/* Effort and Timeframe */}
                  <div className="mt-2 flex items-center gap-2 text-xs text-gray-500">
                    <div className="flex items-center gap-1">
                      <Zap className="w-3 h-3" />
                      <span>{action.effortRequired}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      <span>{action.timeframe.replace('_', ' ')}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Action Summary */}
      <div className="mt-6 bg-gray-50 rounded-lg p-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-gray-900">{data.actionPlan.summary.totalActions}</div>
            <div className="text-sm text-gray-600">Total Actions</div>
          </div>
          
          <div>
            <div className="text-2xl font-bold text-red-600">{data.actionPlan.summary.highPriorityActions}</div>
            <div className="text-sm text-gray-600">High Priority</div>
          </div>
          
          <div>
            <div className="text-2xl font-bold text-green-600">
              +{data.actionPlan.summary.expectedReturnImprovement.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">Expected Return</div>
          </div>
          
          <div>
            <div className="text-2xl font-bold text-blue-600">
              -{data.actionPlan.summary.riskReduction.toFixed(0)}%
            </div>
            <div className="text-sm text-gray-600">Risk Reduction</div>
          </div>
        </div>
      </div>
    </div>
  )
} 