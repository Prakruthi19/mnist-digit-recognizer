import React from 'react';
import './ConfidenceChart.scss';

const ConfidenceChart = ({ predictions, color }) => {
  if (!predictions) return null;

  const sortedPredictions = Object.entries(predictions.all_probabilities)
    .sort((a, b) => parseInt(a[0]) - parseInt(b[0]));

  return (
    <div className="confidence-chart">
      <h3 className="confidence-chart__title">Confidence Distribution</h3>
      <div className="confidence-chart__bars">
        {sortedPredictions.map(([digit, confidence]) => (
          <div key={digit} className="confidence-bar">
            <span className="confidence-bar__label">{digit}</span>
            <div className="confidence-bar__container">
              <div
                className="confidence-bar__fill"
                style={{
                  width: `${confidence * 100}%`,
                  background: color
                }}
              >
                {confidence > 0.1 && (
                  <span className="confidence-bar__percentage">
                    {(confidence * 100).toFixed(1)}%
                  </span>
                )}
              </div>
            </div>
            <span className="confidence-bar__value">
              {(confidence * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ConfidenceChart;