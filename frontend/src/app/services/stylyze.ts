import { HttpClient } from '@angular/common/http';
import { inject, Service } from '@angular/core';

@Service()
export class Stylyze {
  private http = inject(HttpClient);

  stylyze(content: File, style: File) {
    const form = new FormData();
    
    form.append('content', content);
    form.append('style', style);

    return this.http.post('/api/stylyze', form, {
      responseType: 'blob'
    });
  }
}
