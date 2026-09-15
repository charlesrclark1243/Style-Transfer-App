import { HttpErrorResponse } from '@angular/common/http';
import { Component, inject, OnDestroy, signal } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatSliderModule } from '@angular/material/slider';
import { MatSnackBar } from '@angular/material/snack-bar';
import { MatToolbarModule } from '@angular/material/toolbar';
import { firstValueFrom } from 'rxjs';
import { CompareSlider } from './components/compare-slider/compare-slider';
import { ImagePicker } from './components/image-picker/image-picker';
import { STYLE_PRESETS } from './presets';
import { Stylyze } from './services/stylyze';

interface Result {
  before: string;
  after: string;
}

@Component({
  imports: [
    CompareSlider,
    ImagePicker,
    MatButtonModule,
    MatIconModule,
    MatProgressBarModule,
    MatSliderModule,
    MatToolbarModule,
  ],
  selector: 'app-root',
  styleUrl: './app.scss',
  templateUrl: './app.html',
})
export class App implements OnDestroy {
  private readonly stylyzeService = inject(Stylyze);
  private readonly snackBar = inject(MatSnackBar);

  protected readonly stylePresets = STYLE_PRESETS;

  protected readonly content = signal<File | null>(null);
  protected readonly style = signal<File | null>(null);
  protected readonly result = signal<Result | null>(null);
  protected readonly loading = signal(false);

  // Style strength: 1 applies the full style, lower values keep more of the photo. Below 1 is gentler on faces
  protected readonly alpha = signal(0.8);

  protected formatStrength(value: number) {
    return `${Math.round(value * 100)}%`;
  }

  protected async stylyze() {
    const content = this.content();
    const style = this.style();
    if (!content || !style) return;

    this.loading.set(true);
    try {
      const blob = await firstValueFrom(this.stylyzeService.stylyze(content, style, this.alpha()));
      this.revokeResult();
      // Keeps its own URL for the content image, so the comparison still matches after a new one is picked
      this.result.set({ before: URL.createObjectURL(content), after: URL.createObjectURL(blob) });
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
    const result = this.result();
    if (!result) return;
    URL.revokeObjectURL(result.before);
    URL.revokeObjectURL(result.after);
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
