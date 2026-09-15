import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { Stylyze } from './stylyze';

describe('Stylyze', () => {
  let service: Stylyze;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(Stylyze);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should send both images and the style strength', () => {
    const content = new File(['c'], 'content.png', { type: 'image/png' });
    const style = new File(['s'], 'style.jpg', { type: 'image/jpeg' });

    service.stylyze(content, style, 0.6).subscribe();

    const request = http.expectOne('/api/stylyze');
    const body = request.request.body as FormData;
    expect(request.request.method).toBe('POST');
    expect((body.get('content') as File).name).toBe('content.png');
    expect((body.get('style') as File).name).toBe('style.jpg');
    expect(body.get('alpha')).toBe('0.6');
    request.flush(new Blob());
  });
});
