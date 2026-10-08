---
name: angular
description: >-
  Use this skill when the user asks to build, scaffold, modify, or debug an
  Angular frontend application (Angular 2+). Covers components, services,
  routing, state management, forms, dependency injection, testing, and
  Angular CLI commands.
---

# Angular Frontend Skill

## When to Use

Use this skill for any Angular (v2+) web application work: creating components,
services, routes, guards, pipes, state management, forms, and tests.

## Project Setup

### Create a new Angular project
```bash
ng new <project-name> --routing --style=scss --strict
```

### Add dependencies
```bash
ng add @angular/material      # Material Design components
npm install @angular/cdk
npm install rxjs
```

## Core Concepts

### Components
- Class-based components: @Component decorator with selector, templateUrl, styleUrls.
- Inline template: use template: `...` for small components.
- OnPush change detection: use ChangeDetectionStrategy.OnPush and immutable data + AsyncPipe.

```typescript
@Component({
  selector: 'app-user-list',
  templateUrl: './user-list.component.html',
  styleUrls: ['./user-list.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class UserListComponent {
  users$ = this.usersService.getUsers$();
  constructor(private usersService: UsersService) {}
}
```

### Services & Dependency Injection
- Services are plain classes decorated with @Injectable.
- Provide at the right level: providedIn: 'root' for singletons, or providers: [...] in a component.
- Use HttpClient for HTTP; inject it via constructor.

```typescript
@Injectable({providedIn: 'root'})
export class UsersService {
  constructor(private http: HttpClient) {}
  getUsers$(): Observable<User[]> {
    return this.http.get<User[]>('/api/users');
  }
}
```

### Routing
- Define routes in app-routing.module.ts with Routes.
- Use Router.navigate(['/path']) for navigation.
- Guard routes with.CanActivate, CanDeactivate, Resolve.

### State Management
- Prefer Signals (Angular 16+) for local component state: signal, computed, effect.
- For global/cross-component state, use NgRx or RxJS subjects with services.

### Forms
- Template-driven forms: ngModel, ngForm.
- Reactive forms: FormGroup, FormBuilder, FormArray, validators.

### Testing
```bash
ng test          # unit tests (Jasmine + Karma)
ng e2e           # e2e tests (Protractor or WebdriverIO)
```

## Best Practices

1. Use standalone components (standalone: true) instead of modules where possible (Angular 15+).
2. Keep components thin: delegate logic to services, present data only.
3. Use the AsyncPipe in templates to avoid manual subscription management.
4. Lazily load routes with loadChildren / loadComponent.
5. Run ng build --configuration production for production builds; check dist/ output.

## Common Commands

| Command | Purpose |
| --- | --- |
| ng serve | Start dev server |
| ng generate component <name> | Create a component |
| ng generate service <name> | Create a service |
| ng generate module <name> --routing | Create a routing module |
| ng build | Build the project |
| ng test | Run unit tests |
| ng lint | Run linter |
