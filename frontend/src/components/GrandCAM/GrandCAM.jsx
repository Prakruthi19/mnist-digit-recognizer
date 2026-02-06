import React from 'react';
import { TrendingUp } from 'lucide-react';
import './GrandCAM.scss';

const GradCAM = ({ gradcamImage }) => {
  if (!gradcamImage) return null;

  return (
    <div className="gradcam">
      <div className="gradcam__header">
        <TrendingUp className="gradcam__icon" />
        <h3 className="gradcam__title">🔬 Grad-CAM Heatmap</h3>
      </div>
      <p className="gradcam__description">What the CNN focuses on:</p>
      <div className="gradcam__image-container">
        <img 
          src={gradcamImage} 
          alt="Grad-CAM Heatmap" 
          className="gradcam__image"
        />
      </div>
    </div>
  );
};

export default GradCAM;