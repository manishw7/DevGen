import { Routes, Route, NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  ScanLine,
  Database,
  BarChart3,
  Sparkles,
  Settings,
  BookOpen,
} from 'lucide-react';

import Dashboard from './pages/Dashboard';
import Recognize from './pages/Recognize';
import DatasetExplorer from './pages/DatasetExplorer';
import Evaluate from './pages/Evaluate';
import NERPage from './pages/NERPage';
import GenerativeSuite from './pages/GenerativeSuite';

function App() {

  const navItems = [
    { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/recognize', icon: ScanLine, label: 'OCR Recognition' },
    { path: '/dataset', icon: Database, label: 'Dataset Explorer' },
    { path: '/evaluate', icon: BarChart3, label: 'CER Evaluation' },
    { path: '/ner', icon: BookOpen, label: 'NER Extraction' },
    { path: '/generate', icon: Sparkles, label: 'Generative Suite' },
  ];

  const futureItems = [
    { icon: Settings, label: 'Settings' },
  ];

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <div className="sidebar-logo-icon">D</div>
            <div>
              <div className="sidebar-logo-text">DevGen</div>
              <div className="sidebar-subtitle">Devanagari OCR Suite</div>
            </div>
          </div>
        </div>

        <nav className="sidebar-nav">
          <div className="sidebar-section-title">Main</div>
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `nav-link ${isActive ? 'active' : ''}`
              }
              end={item.path === '/'}
            >
              <item.icon />
              {item.label}
            </NavLink>
          ))}

          <div className="sidebar-section-title" style={{ marginTop: '8px' }}>
            Coming Soon
          </div>
          {futureItems.map((item) => (
            <div
              key={item.label}
              className="nav-link"
              style={{ opacity: 0.4, cursor: 'not-allowed' }}
            >
              <item.icon />
              {item.label}
            </div>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-footer-badge">
            <span className="dot" />
            TrOCR Engine Ready
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/recognize" element={<Recognize />} />
          <Route path="/dataset" element={<DatasetExplorer />} />
          <Route path="/evaluate" element={<Evaluate />} />
          <Route path="/ner" element={<NERPage />} />
          <Route path="/generate" element={<GenerativeSuite />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
