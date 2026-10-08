import React from 'react';
import { Inbox } from 'lucide-react';

interface EmptyStateProps {
  icon?: React.ElementType;
  title: string;
  description?: string;
  action?: { label: string; onClick: () => void };
  className?: string;
}

/**
 * 统一空状态。
 * 此前有两套不一致实现：MiningDashboardPage 是整页居中块，FactorList 是表格里一行灰字，
 * 样式与文案密度都不统一 —— 统一收敛到这里。
 */
export const EmptyState: React.FC<EmptyStateProps> = ({
  icon: Icon = Inbox,
  title,
  description,
  action,
  className = '',
}) => (
  <div className={`flex flex-col items-center justify-center py-12 text-center animate-fade-in-up ${className}`}>
    <div className="mb-4 p-4 rounded-2xl bg-secondary/40 border border-border/50">
      <Icon className="h-8 w-8 text-muted-foreground" />
    </div>
    <p className="text-sm font-medium text-foreground/80">{title}</p>
    {description && (
      <p className="mt-1.5 text-xs text-muted-foreground max-w-sm leading-relaxed">{description}</p>
    )}
    {action && (
      <button onClick={action.onClick} className="mt-4 text-sm text-primary hover:underline">
        {action.label}
      </button>
    )}
  </div>
);
