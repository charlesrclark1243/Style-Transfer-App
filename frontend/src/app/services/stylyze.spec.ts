import { TestBed } from '@angular/core/testing';
import { Stylyze } from './stylyze';

describe('Stylyze', () => {
  let service: Stylyze;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(Stylyze);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
