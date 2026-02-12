import { Sidebar } from './sidebar'
import { Header } from './header'
import { Footer } from './footer'

export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="relative min-h-screen">
      <Sidebar />
      <div className="flex min-h-screen flex-1 flex-col md:pl-[252px]">
        <Header />
        <main className="flex-1 overflow-y-auto px-4 pb-8 pt-5 md:px-8 md:pt-7">
          <div className="mx-auto w-full max-w-7xl animate-fade-in">
            {children}
          </div>
        </main>
        <Footer />
      </div>
    </div>
  )
}
