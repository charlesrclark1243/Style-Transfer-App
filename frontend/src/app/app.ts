import { HttpErrorResponse } from '@angular/common/http';
import { Component, inject, OnDestroy, signal } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatSnackBar } from '@angular/material/snack-bar';
import { firstValueFrom } from 'rxjs';
import { ImagePicker } from './components/image-picker/image-picker';
import { Stylyze } from './services/stylyze';

@Component({
  imports: [ImagePicker, MatButtonModule, MatProgressBarModule],
  selector: 'app-root',
  styleUrl: './app.scss',
  templateUrl: './app.html',
})
export class App implements OnDestroy {
  private readonly stylyzeService = inject(Stylyze);
  private readonly snackBar = inject(MatSnackBar);

  protected readonly content = signal<File | null>(null);
  protected readonly style = signal<File | null>(null);
  protected readonly resultUrl = signal<string | null>(null);
  protected readonly loading = signal(false);

  protected async stylyze() {
    const content = this.content();
    const style = this.style();
    if (!content || !style) return;

    this.loading.set(true);
    try {
      const blob = await firstValueFrom(this.stylyzeService.stylyze(content, style));
      this.revokeResult();
      this.resultUrl.set(URL.createObjectURL(blob));
    } catch (err) {
      this.snackBar.open(await errorMessage(err), 'Dismiss', { duration: 6000 });
    } finally {
      this.loading.set(false);
    }
  }

  ngOnDestroy() {
    this.revokeResult();
  }

  private revokeResult() {
    const url = this.resultUrl();
    if (url) URL.revokeObjectURL(url);
  }
}

// With responseType 'blob', error bodies arrive as a Blob too, so FastAPI's JSON detail has to be read out of it
async function errorMessage(err: unknown): Promise<string> {
  if (err instanceof HttpErrorResponse && err.error instanceof Blob) {
    try {
      const body = JSON.parse(await err.error.text());
      if (typeof body.detail === 'string') return body.detail;
    } catch {
      // Not JSON, e.g. an nginx error page
    }
  }
  return 'Something went wrong. Please try again.';
}
