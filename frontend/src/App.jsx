import React, { useState } from 'react';
import { Brain, Cpu } from 'lucide-react';
import Header from '../src/components/Header/Header';
import DrawingCanvas from '../src/components/DrawingCanvas/DrawingCanvas';
import PredictionDisplay from '../src/components/PredictionDisplay/PredictionDisplay';
import GradCAM from '../src/components/GrandCAM/GrandCAM';
import ComparisonSection from '../src/components/ComparisonSection/ComparisonSection';
import Footer from '../src/components/Footer/Footer';
import './App.scss';

const App = () => {
  const [mlpPrediction, setMlpPrediction] = useState(null);
  const [cnnPrediction, setCnnPrediction] = useState(null);
  const [gradcamImage, setGradcamImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handlePredict = async (imageData) => {
    setIsLoading(true);
    
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData })
      });
      
      if (!response.ok) {
        throw new Error('Prediction failed');
      }
      
      const result = await response.json();
      
      setMlpPrediction(result.mlp);
      setCnnPrediction(result.cnn);
      setGradcamImage(result.gradcam);
      
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to predict. Make sure the backend server is running on http://localhost:5000');
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setMlpPrediction(null);
    setCnnPrediction(null);
    setGradcamImage(null);
  };

  return (
    <div className="app">
      <div className="app__container">
        <Header />
        
        <main className="app__main">
          <div className="app__grid">
            
            {/* Drawing Section */}
            <div className="app__section">
              <DrawingCanvas
                onPredict={handlePredict}
                onClear={handleClear}
                isLoading={isLoading}
              />
            </div>

            {/* MLP Results */}
            <div className="app__section">
              <PredictionDisplay
                prediction={mlpPrediction}
                icon={Brain}
                title="MLP (Simple Neural Network)"
                color="#ef4444"
                modelType="mlp"
              />
            </div>

            {/* CNN Results */}
            <div className="app__section">
              <PredictionDisplay
                prediction={cnnPrediction}
                icon={Cpu}
                title="CNN (Convolutional Network)"
                color="#10b981"
                modelType="cnn"
              />
              
              <GradCAM gradcamImage={gradcamImage} />
            </div>

          </div>
        </main>

        <ComparisonSection 
          mlpPrediction={mlpPrediction} 
          cnnPrediction={cnnPrediction} 
        />

        <Footer />
      </div>
    </div>
  );
};

export default App;