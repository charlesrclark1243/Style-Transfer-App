import { HttpClient } from "@angular/common/http";
import { inject, Service } from "@angular/core";

@Service()
export class Stylyze {
  private http = inject(HttpClient);

  stylyze(content: File, style: File, alpha: number) {
    const form = new FormData();

    form.append("content", content); // names must match the backend
    form.append("style", style);
    form.append("alpha", String(alpha));

    return this.http.post("/api/stylyze", form, {
      responseType: "blob"
    });
  }
}