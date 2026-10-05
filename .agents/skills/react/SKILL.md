---
name: react
description: >-
  Use this skill when the user asks to build, scaffold, modify, or debug a
  React frontend application. Covers components, hooks, state management,
  routing, forms, testing, performance optimization, and modern React patterns
  (React 18+).
---

# React Frontend Skill

## When to Use

Use this skill for any React (v18+) web application work: creating components,
hooks, state management, routing, forms, testing, and performance tuning.

## Project Setup

### Create a new React project (Vite recommended)
```bash
# Vite + React + TypeScript
npm create vite@latest my-app -- --template react-ts

# Or CRA (Create React App)
npx create-react-app my-app --template typescript
```

### Key Dependencies
```bash
npm install react react-dom
npm install react-router-dom        # client-side routing
npm install @tanstack/react-query    # data fetching / caching
npm install zod                      # schema validation
npm install zustand                   # lightweight global state
npm install tailwindcss               # styling (optional)
```

## Core Concepts

### Components & Hooks (React 18+)
- Use **function components** with hooks. Avoid class components.
- **useState**, **useEffect**, **useMemo**, **useCallback**, **useRef**.
- **useSyncExternalStore** for external store subscriptions.
- Custom hooks to extract reusable logic.

```tsx
interface User { id: number; name: string; }

export function UserList() {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchUsers = useCallback(async () => {
    setLoading(true);
    const res = await fetch('/api/users');
    const data = await res.json() as User[];
    setUsers(data);
    setLoading(false);
  }, []);

  useEffect(() => { void fetchUsers(); }, [fetchUsers]);

  if (loading) return <p>Loading...</p>;
  return <ul>{users.map(u => <li key={u.id}>{u.name}</li>)}</ul>;
}
```

### State Management
- **Local state**: `useState` / `useReducer`.
- **Server state**: `@tanstack/react-query` or SWR.
- **Global UI state**: Zustand, Jotai, or Redux Toolkit (only when truly global).
- Avoid lifting state too high; keep it as close as possible.

### Routing (React Router v6+)
```tsx
import { createBrowserRouter, RouterProvider, Route, Outlet } from 'react-router-dom';

const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      { path: '', element: <Home /> },
      { path: 'users', element: <Users /> },
      { path: 'users/:id', element: <UserDetail /> },
    ],
  },
]);
```

### Data Fetching
- Prefer `@tanstack/react-query` over manual `useEffect` + `fetch`.
- Use suspense boundaries for declarative loading states.

### Forms
- Use `react-hook-form` + `@hookform/resolvers` (zod) for form validation.
- Avoid uncontrolled inputs in complex forms.

### Testing
```bash
npm install --save-dev vitest jsdom @testing-library/react @testing-library/jest-dom
npm install --save-dev @testing-library/user-event
```

```tsx
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect } from 'vitest';

describe('UserList', () => {
  it('renders users', () => {
    render(<UserList />);
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });
});
```

## Best Practices

1. **React 18+ concurrency**: Use `useTransition`, `useDeferredValue` for heavy UI.
2. **Key prop**: Always provide stable keys in lists.
3. **Avoid inline objects/arrays** in props or `useEffect` deps to prevent re-renders.
4. **Code splitting**: Use `React.lazy` + `Suspense` for route-level lazy loading.
5. **Type everything**: Use TypeScript with strict mode.
6. **Accessibility**: Use semantic HTML, `aria-*` attributes, keyboard support.
7. **Production build**: `npm run build` (Vite) or `CI=true npm run build` (CRA).

## Common Commands

| Command | Purpose |
| --- | --- |
| `npm run dev` | Start dev server |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm test` | Run tests |
| `npm run lint` | Run linter |
