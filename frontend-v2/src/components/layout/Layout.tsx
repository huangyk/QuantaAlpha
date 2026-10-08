import React from 'react';
import { Sparkles, Database, BarChart3, Settings as SettingsIcon } from 'lucide-react';
import { useTaskContext } from '@/context/TaskContext';

export type PageId = 'home' | 'library' | 'backtest' | 'settings' | 'mining_dashboard';

interface LayoutProps {
  children: React.ReactNode;
  currentPage: PageId;
  onNavigate: (page: PageId) => void;
  showNavigation?: boolean;
}

export const Layout: React.FC<LayoutProps> = ({
  children,
  currentPage,
  onNavigate,
  showNavigation = true,
}) => {
  const { miningTask } = useTaskContext();

  // Helper to determine where 'Factor Mining' nav item should go
  const handleNavClick = (itemId: PageId) => {
    if (itemId === 'home') {
      // If there is an active mining task (running, completed, or failed), go to dashboard
      // Only go to home welcome screen if task is null or idle
      if (miningTask && miningTask.status !== 'idle') {
        onNavigate('mining_dashboard');
      } else {
        onNavigate('home');
      }
    } else {
      onNavigate(itemId);
    }
  };

  const navItems = [
    { id: 'home' as const, label: '因子挖掘', icon: Sparkles },
    { id: 'library' as const, label: '因子库', icon: Database },
    { id: 'backtest' as const, label: '回测', icon: BarChart3 },
    { id: 'settings' as const, label: '设置', icon: SettingsIcon },
  ];

  return (
    <div className="min-h-screen bg-background gradient-mesh">
      {/* ★ 左侧导航栏（原为顶部横栏）：窄屏只显示图标 w-[72px]，lg 以上展开为 w-64 */}
      <aside className="fixed left-0 top-0 bottom-0 z-40 w-[72px] lg:w-64 glass-strong border-r border-border/50 flex flex-col">
        {/* Logo */}
        <div
          className="flex items-center gap-3 px-3 lg:px-5 py-5 border-b border-border/40 cursor-pointer hover:opacity-80 transition-opacity"
          onClick={() => onNavigate('home')}
        >
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-purple-600">
            <Sparkles className="h-6 w-6 text-white" />
          </div>
          <div className="hidden lg:block min-w-0">
            <h1 className="text-lg font-bold truncate">QuantaAlpha</h1>
            <p className="text-xs text-muted-foreground truncate">智能因子挖掘平台</p>
          </div>
        </div>

        {/* Navigation */}
        {showNavigation && (
          <nav className="flex-1 flex flex-col gap-1 px-2 lg:px-3 py-4 overflow-y-auto">
            {navItems.map((item) => {
              const Icon = item.icon;
              // Highlight 'Factor Mining' if we are on dashboard OR home
              const isActive = currentPage === item.id || (item.id === 'home' && currentPage === 'mining_dashboard');
              return (
                <button
                  key={item.id}
                  onClick={() => handleNavClick(item.id)}
                  title={item.label}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all justify-center lg:justify-start ${
                    isActive
                      ? 'bg-primary text-primary-foreground shadow-[0_2px_10px_rgba(139,124,246,0.25)]'
                      : 'text-muted-foreground hover:text-foreground hover:bg-secondary/50'
                  }`}
                >
                  <Icon className="h-5 w-5 shrink-0" />
                  <span className="hidden lg:inline text-sm font-medium whitespace-nowrap">{item.label}</span>
                </button>
              );
            })}
          </nav>
        )}
      </aside>

      {/* Main Content — pb-48 ensures content is never hidden behind the fixed ChatInput */}
      <main className="ml-[72px] lg:ml-64 pt-8 pb-48">
        <div className="container mx-auto px-6">{children}</div>
      </main>
    </div>
  );
};
