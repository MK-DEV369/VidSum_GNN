import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import HomePage from './pages/HomePage'
import DashboardPage from './pages/DashboardPage'
import './index.css'

function App() {

  return (
    <Router>
      <div className="min-h-screen flex flex-col">
        {/* Main Content */}
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
          </Routes>
        </main>

        {/* Footer */}
        <footer className="border-t bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 text-center py-8 text-sm text-slate-300 backdrop-blur-sm">
          <div className="max-w-6xl mx-auto">
            <p className="font-medium">VIDSUM-GNN © 2026 - AI253IA: Artificial Neural Networks and Deep Learning</p>
          </div>
        </footer>
      </div>
    </Router>
  )
}

export default App
