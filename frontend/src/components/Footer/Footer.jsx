import React from 'react';
import './Footer.scss';

const Footer = () => {
  return (
    <footer className="footer">
      <p className="footer__text">
        Built with TensorFlow, Keras & Flask | 
        <a 
          href="https://github.com/yourusername/mnist-digit-recognizer" 
          target="_blank" 
          rel="noopener noreferrer"
          className="footer__link"
        >
          View on GitHub
        </a>
      </p>
      <p className="footer__tech">
        🧠 Deep Learning • 🎨 React • ⚡ Real-time Predictions
      </p>
    </footer>
  );
};

export default Footer;